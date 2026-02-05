# -*- coding: utf-8 -*-
"""
NIHL 队列数据分析示例

本示例演示如何使用 ohtk 工具包进行职业性噪声聋（NIHL）的
队列数据分析，包括：
1. 通过 StaffInfo 对象加载数据（OOP 方式）
2. GEE 模型拟合
3. LightGBM 预测模型
4. Cox 比例风险模型
5. 阈值分析
6. 分位数回归分析

数据文件：
    将队列数据文件放置在 examples/data/ 目录下
    默认文件名: lex_aggregated_by_report_median_queue_data.csv
"""

import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

# 导入 ohtk 模块
from ohtk.staff_info import StaffInfo
from ohtk.modeling.statistical import GEEModel, fit_gee, CoxPHModel, fit_cox
from ohtk.modeling.boosting import LGBMNIHLPredictor, train_lgbm_nihl
from ohtk.algorithms.analysis import (
    find_optimal_threshold,
    analyze_nonlinearity,
    PiecewiseThresholdFinder,
    QuantileRegressionAnalyzer,
)


def setup_paths(args):
    """设置路径"""
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    models_path = Path(args.models_path)
    
    for path in (output_path, models_path):
        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"创建目录: {path}")
    
    return input_path, output_path, models_path


def load_raw_dataframe(input_path: Path, filename: str) -> pd.DataFrame:
    """
    加载原始 CSV 数据到 DataFrame
    
    Args:
        input_path: 输入路径
        filename: 文件名
        
    Returns:
        原始 DataFrame
    """
    file_path = input_path / filename
    
    if not file_path.exists():
        logger.error(f"数据文件不存在: {file_path}")
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"加载原始数据: {len(df)} 行, {len(df.columns)} 列")
    
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理 DataFrame（基础清洗，不计算队列特征）
    
    注意：check_order 和 days_since_first 将通过 StaffInfo.build_queue_features() 计算
    
    Args:
        df: 原始 DataFrame
        
    Returns:
        预处理后的 DataFrame
    """
    df = df.copy()
    
    # 转换日期
    if "creation_date" in df.columns:
        df["creation_date"] = pd.to_datetime(df["creation_date"])
    
    # 按工人和日期排序
    df = df.sort_values(["worker_id", "creation_date"])
    
    # 删除含有关键字段缺失值的行
    # 注意：此处不要求 NIHL 字段，因为 NIHL 可以通过 StaffInfo 计算
    required_cols = ["worker_id", "sex", "age", "LEX_8h_LEX_40h_median"]
    existing_cols = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_cols)
    df = df.reset_index(drop=True)
    
    logger.info(f"预处理后数据: {len(df)} 行")
    
    return df


def load_staff_info_objects(df: pd.DataFrame) -> Dict[str, StaffInfo]:
    """
    通过 StaffInfo 对象加载数据（OOP 方式）
    
    展示 ohtk 的面向对象设计：
    - 每个工人对应一个 StaffInfo 对象
    - StaffInfo 包含时间序列健康数据
    - 队列特征通过 build_queue_features() 显式计算
    
    Args:
        df: 预处理后的 DataFrame
        
    Returns:
        worker_id -> StaffInfo 的字典
    """
    logger.info("=" * 60)
    logger.info("步骤 1: 通过 StaffInfo 对象加载数据")
    logger.info("=" * 60)
    
    # 使用 StaffInfo 的批量加载方法
    staff_dict = StaffInfo.load_batch_from_dataframe(df)
    
    logger.info(f"成功加载 {len(staff_dict)} 个工人的 StaffInfo 对象")
    
    # 展示 StaffInfo 对象的信息
    if staff_dict:
        sample_id = list(staff_dict.keys())[0]
        sample_staff = staff_dict[sample_id]
        
        logger.info(f"\n示例 StaffInfo 对象 (worker_id={sample_id}):")
        logger.info(f"  - 性别: {sample_staff.staff_sex}")
        logger.info(f"  - 年龄: {sample_staff.staff_age}")
        logger.info(f"  - 工龄: {sample_staff.staff_duration}")
        logger.info(f"  - 检查次数: {len(sample_staff.record_dates)}")
        logger.info(f"  - check_order: {sample_staff.check_order} (未计算)")
        logger.info(f"  - days_since_first: {sample_staff.days_since_first} (未计算)")
        
        # 展示时间序列数据
        if sample_staff.record_dates:
            logger.info(f"  - 检查日期范围: {min(sample_staff.record_dates)} ~ {max(sample_staff.record_dates)}")
            
            # 展示某次检查的健康信息
            recent_date = sample_staff.get_most_recent_date()
            if recent_date and recent_date in sample_staff.staff_health_info:
                health_info = sample_staff.staff_health_info[recent_date]
                if health_info.auditory:
                    logger.info(f"  - 最近一次 NIHL: {health_info.auditory.NIHL}")
    
    return staff_dict


def build_queue_features(staff_dict: Dict[str, StaffInfo]) -> None:
    """
    步骤 2: 计算队列特征（check_order, days_since_first）
    
    Args:
        staff_dict: worker_id -> StaffInfo 的字典
    """
    logger.info("=" * 60)
    logger.info("步骤 2: 计算队列特征")
    logger.info("=" * 60)
    
    # 使用 StaffInfo 的批量方法计算队列特征
    StaffInfo.build_queue_features_batch(staff_dict)
    
    # 展示计算结果
    if staff_dict:
        sample_id = list(staff_dict.keys())[0]
        sample_staff = staff_dict[sample_id]
        
        logger.info(f"\n计算后的 StaffInfo (worker_id={sample_id}):")
        logger.info(f"  - check_order: {dict(sample_staff.check_order)}")
        logger.info(f"  - days_since_first: {dict(sample_staff.days_since_first)}")


def staff_dict_to_analysis_dataframe(
    staff_dict: Dict[str, StaffInfo]
) -> pd.DataFrame:
    """
    将 StaffInfo 对象字典转换为分析用 DataFrame
    
    这是将 OOP 数据结构转换为统计建模所需格式的关键步骤
    
    注意：此函数已被弃用，请使用 StaffInfo.to_analysis_dataframe_batch()
    
    Args:
        staff_dict: worker_id -> StaffInfo 的字典
        
    Returns:
        适合统计建模的 DataFrame
    """
    logger.info("=" * 60)
    logger.info("步骤 3: 转换为分析 DataFrame")
    logger.info("=" * 60)
    
    # 使用 StaffInfo 的新方法进行转换
    df = StaffInfo.to_analysis_dataframe_batch(staff_dict)
    
    # 重命名 LAeq 列（如果需要兼容旧代码）
    if "LAeq" not in df.columns and "LEX_8h_LEX_40h_median" in df.columns:
        df["LAeq"] = df["LEX_8h_LEX_40h_median"]
    
    # 删除缺失值
    df = df.dropna(subset=["NIHL346", "LAeq"])
    df = df.reset_index(drop=True)
    
    logger.info(f"转换后 DataFrame: {len(df)} 行, {df['worker_id'].nunique()} 个工人")
    
    # 展示 NIHL 标签转换功能
    if "NIHL1234" in df.columns and "NIHL346" in df.columns:
        from ohtk.utils.pta_correction import convert_nihl_to_labels
        
        logger.info("\n展示 NIHL 标签转换功能:")
        df["norm_hearing_loss_label"] = convert_nihl_to_labels(df["NIHL1234"], encoding="categorical")
        df["high_hearing_loss_label"] = convert_nihl_to_labels(df["NIHL346"], encoding="categorical")
        
        logger.info(f"  - norm_hearing_loss 分布:\n{df['norm_hearing_loss_label'].value_counts()}")
        logger.info(f"  - high_hearing_loss 分布:\n{df['high_hearing_loss_label'].value_counts()}")
    
    return df


def demonstrate_staff_info_methods(staff_dict: Dict[str, StaffInfo]):
    """
    展示 StaffInfo 对象的方法调用
    
    Args:
        staff_dict: worker_id -> StaffInfo 的字典
    """
    logger.info("=" * 60)
    logger.info("展示 StaffInfo 对象方法")
    logger.info("=" * 60)
    
    if not staff_dict:
        logger.warning("没有可用的 StaffInfo 对象")
        return
    
    # 选择一个有完整数据的工人
    sample_staff = None
    for staff in staff_dict.values():
        if (staff.staff_age and staff.staff_sex and 
            staff.staff_duration and staff.staff_occhaz_info):
            sample_staff = staff
            break
    
    if sample_staff is None:
        sample_staff = list(staff_dict.values())[0]
    
    logger.info(f"使用工人 {sample_staff.staff_id} 演示方法调用")
    
    # 尝试 NIPTS 预测
    try:
        # 获取 LAeq
        recent_key = sample_staff.get_record_key()
        laeq = None
        if recent_key and recent_key in sample_staff.staff_occhaz_info:
            occhaz = sample_staff.staff_occhaz_info[recent_key]
            if occhaz.noise_hazard_info:
                if isinstance(occhaz.noise_hazard_info, dict):
                    laeq = occhaz.noise_hazard_info.get("LAeq")
        
        if laeq and sample_staff.staff_age and sample_staff.staff_duration:
            # ISO 1999:2013 预测
            nipts_2013 = sample_staff.NIPTS_predict_iso1999_2013(
                LAeq=laeq,
                percentrage=50,
                mean_key=[3000, 4000, 6000]
            )
            logger.info(f"  ISO 1999:2013 NIPTS 预测: {nipts_2013:.2f} dB")
            
            # ISO 1999:2023 预测
            nipts_2023 = sample_staff.NIPTS_predict_iso1999_2023(
                LAeq=laeq,
                percentrage=50,
                mean_key=[3000, 4000, 6000]
            )
            logger.info(f"  ISO 1999:2023 NIPTS 预测: {nipts_2023:.2f} dB")
        else:
            logger.warning("  缺少 NIPTS 预测所需数据")
    except Exception as e:
        logger.warning(f"  NIPTS 预测失败: {e}")
    
    # 尝试 NIHL 计算
    try:
        nihl_result = sample_staff.calculate_auditory_nihl(
            freq_keys=["1234", "346"],
            apply_correction=False
        )
        logger.info(f"  NIHL 计算结果: {nihl_result}")
    except Exception as e:
        logger.warning(f"  NIHL 计算失败: {e}")


def run_gee_analysis(df: pd.DataFrame):
    """
    运行 GEE（广义估计方程）分析
    
    GEE 适用于纵向数据分析，考虑个体内重复测量的相关性
    """
    logger.info("=" * 60)
    logger.info("步骤 2: GEE 模型分析")
    logger.info("=" * 60)
    
    gee = GEEModel(
        outcome="NIHL346",
        exposure="LAeq",
        covariates=["check_order", "sex", "age"],
        group_var="worker_id"
    )
    
    # 自动选择最优相关结构
    result = gee.fit(df, auto_select_corr=True)
    
    logger.info(f"最优相关结构: {result['best_structure']}")
    logger.info(f"暴露效应系数: {result['best_result']['exposure_coef']:.4f}")
    logger.info(f"暴露效应 p 值: {result['best_result']['exposure_p']:.4f}")
    
    logger.info("\n模型摘要:")
    logger.info(gee.summary())
    
    return result


def run_lgbm_analysis(df: pd.DataFrame, models_path: Path):
    """
    运行 LightGBM 预测模型分析
    """
    logger.info("=" * 60)
    logger.info("步骤 3: LightGBM 预测模型")
    logger.info("=" * 60)
    
    predictor = LGBMNIHLPredictor(
        outcome="NIHL346",
        exposure="LAeq",
        base_features=["check_order", "age", "sex", "days_since_first"],
        group_var="worker_id"
    )
    
    # 特征工程
    logger.info("执行特征工程...")
    df_engineered = predictor.engineer_features(
        df,
        add_interactions=True,
        add_polynomial=True,
        add_lags=True,
        lag_features=["NIHL346", "LAeq"],
        lags=[1, 2]
    )
    
    # 训练模型
    logger.info("训练 LightGBM 模型...")
    result = predictor.fit(
        df_engineered,
        test_size=0.2,
        num_boost_round=1000,
        early_stopping_rounds=50
    )
    
    logger.info(f"测试集 RMSE: {result['rmse']:.4f}")
    logger.info(f"测试集 R²: {result['r2']:.4f}")
    logger.info(f"最佳迭代次数: {result['best_iteration']}")
    
    logger.info("\n特征重要性排名 (前10):")
    logger.info(result['feature_importance'].head(10).to_string())
    
    # 保存模型
    model_file = models_path / "queue-lgbm-model.txt"
    predictor.save(model_file)
    logger.info(f"模型已保存到: {model_file}")
    
    return result


def run_cox_analysis(df: pd.DataFrame):
    """
    运行 Cox 比例风险模型分析
    """
    logger.info("=" * 60)
    logger.info("步骤 4: Cox 比例风险模型")
    logger.info("=" * 60)
    
    cox = CoxPHModel(
        outcome="NIHL346",
        exposure="LAeq",
        covariates=["age", "sex"],
        threshold=25.0,
        group_var="worker_id",
        time_var="check_order"
    )
    
    try:
        result = cox.fit(df)
        
        logger.info(f"一致性指数 (C-index): {result['c_index']:.4f}")
        logger.info(f"事件数: {result['n_events']}/{result['n_samples']}")
        logger.info(f"暴露风险比 (HR): {result['exposure_hr']:.4f}")
        logger.info(f"暴露 p 值: {result['exposure_p']:.4f}")
        
        logger.info("\n模型摘要:")
        logger.info(cox.summary())
        
        return result
    except ImportError:
        logger.warning("lifelines 未安装，跳过 Cox 分析")
        return None


def run_threshold_analysis(df: pd.DataFrame):
    """
    运行阈值分析
    """
    logger.info("=" * 60)
    logger.info("步骤 5: 阈值分析")
    logger.info("=" * 60)
    
    finder = PiecewiseThresholdFinder(
        outcome="NIHL346",
        exposure="LAeq",
        covariates=["check_order", "sex", "age"]
    )
    
    threshold, results = finder.find_threshold(
        df,
        min_percentile=10,
        max_percentile=90,
        step=5
    )
    
    if threshold is not None:
        logger.info(f"最佳阈值: {threshold:.2f} dB")
        logger.info("\n阈值分析详细结果 (前10):")
        logger.info(results.head(10).to_string())
    
    return threshold, results


def run_quantile_regression(df: pd.DataFrame):
    """
    运行分位数回归分析
    """
    logger.info("=" * 60)
    logger.info("步骤 6: 分位数回归分析")
    logger.info("=" * 60)
    
    analyzer = QuantileRegressionAnalyzer(
        outcome="NIHL346",
        exposure="LAeq",
        covariates=["check_order", "sex", "age"]
    )
    
    result = analyzer.analyze(
        df,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    
    logger.info(f"效应变异系数: {result['effect_variation_coefficient']:.4f}")
    logger.info("\n各分位数效应:")
    logger.info(result['quantile_effects'].to_string())
    
    return result


def main():
    """主函数"""
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(
        description="NIHL 队列数据分析示例（StaffInfo OOP 方式）"
    )
    parser.add_argument(
        "--input_path", type=str,
        default="./data",
        help="输入数据路径"
    )
    parser.add_argument(
        "--output_path", type=str,
        default="./results",
        help="输出结果路径"
    )
    parser.add_argument(
        "--models_path", type=str,
        default="./models",
        help="模型保存路径"
    )
    parser.add_argument(
        "--filename", type=str,
        default="lex_aggregated_by_report_median_queue_data.csv",
        help="数据文件名"
    )
    parser.add_argument(
        "--task", type=str,
        default="all",
        choices=["all", "load", "gee", "lgbm", "cox", "threshold", "quantile"],
        help="要执行的分析任务"
    )
    
    args = parser.parse_args()
    
    # 打印参数
    logger.info("NIHL 队列数据分析示例（StaffInfo OOP 方式）")
    logger.info("=" * 60)
    logger.info("输入参数:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    
    # 设置路径
    input_path, output_path, models_path = setup_paths(args)
    
    # ========================================
    # 数据流：加载 -> 队列聚合 -> 转换 DataFrame
    # ========================================
    
    # 步骤 1: 加载原始 CSV
    try:
        raw_df = load_raw_dataframe(input_path, args.filename)
    except FileNotFoundError:
        return
    
    # 步骤 2: 基础预处理（不计算队列特征）
    df = preprocess_dataframe(raw_df)
    
    # 步骤 3: 通过 StaffInfo 对象加载（OOP 方式）
    staff_dict = load_staff_info_objects(df)
    
    # 步骤 4: 计算队列特征（check_order, days_since_first）
    build_queue_features(staff_dict)
    
    # 步骤 5: 展示 StaffInfo 方法
    demonstrate_staff_info_methods(staff_dict)
    
    # 步骤 6: 转换回 DataFrame 用于统计建模
    analysis_df = staff_dict_to_analysis_dataframe(staff_dict)
    
    if args.task == "load":
        logger.info("数据加载完成，退出")
        return
    
    # ========================================
    # 步骤 7+: 统计分析
    # ========================================
    
    task = args.task
    
    if task in ["all", "gee"]:
        try:
            run_gee_analysis(analysis_df)
        except Exception as e:
            logger.error(f"GEE 分析失败: {e}")
    
    if task in ["all", "lgbm"]:
        try:
            run_lgbm_analysis(analysis_df, models_path)
        except Exception as e:
            logger.error(f"LightGBM 分析失败: {e}")
    
    if task in ["all", "cox"]:
        try:
            run_cox_analysis(analysis_df)
        except Exception as e:
            logger.error(f"Cox 分析失败: {e}")
    
    if task in ["all", "threshold"]:
        try:
            run_threshold_analysis(analysis_df)
        except Exception as e:
            logger.error(f"阈值分析失败: {e}")
    
    if task in ["all", "quantile"]:
        try:
            run_quantile_regression(analysis_df)
        except Exception as e:
            logger.error(f"分位数回归失败: {e}")
    
    logger.info("=" * 60)
    logger.info("分析完成!")


if __name__ == "__main__":
    main()
