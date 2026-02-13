# -*- coding: utf-8 -*-
"""
实验基类模块

提供标准化的实验流程框架，支持回归、分类、聚类任务扩展
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import json

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, explained_variance_score, median_absolute_error
)

from ohtk.experiments.feature_builder import FeatureBuilder
from ohtk.experiments.model_trainer import CrossValidationTrainer, ModelWrapper
from ohtk.staff_info import StaffInfo
from ohtk.diagnose_info.auditory_diagnose import AuditoryDiagnose


class BaseExperiment(ABC):
    """
    实验抽象基类
    
    支持回归、分类、聚类任务扩展
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化实验
        
        Args:
            config: 实验配置字典
        """
        self.config = config
        self.results = {}
        self.data = None
        
        # 提取常用配置
        data_file_path = config.get('data', {}).get('file_path', '')
        self.data_file = Path(data_file_path)
        
        # 如果是相对路径，转换为绝对路径（相对于当前工作目录或配置文件位置）
        if not self.data_file.is_absolute():
            # 尝试相对于当前工作目录
            if not self.data_file.exists():
                # 尝试相对于项目根目录（假设配置文件在项目中）
                project_root = Path.cwd()
                # 向上查找直到找到 examples 目录
                for _ in range(5):  # 最多向上5层
                    if (project_root / data_file_path).exists():
                        self.data_file = project_root / data_file_path
                        break
                    project_root = project_root.parent
        
        self.output_dir = Path(config.get('output', {}).get('dir', 'results'))
        self.experiment_name = config.get('experiment_name', 'experiment')
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / f"{self.experiment_name}.log"
        logger.add(log_file, encoding='utf-8')
        
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        执行实验，子类必须实现
        
        Returns:
            实验结果字典
        """
        pass
    
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        评估指标计算，子类必须实现
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估指标字典
        """
        pass
    
    def save_results(self):
        """保存详细结果"""
        results_file = self.output_dir / f"{self.experiment_name}_results.json"
        
        # 转换结果为可序列化格式
        serializable_results = {}
        for target, result in self.results.items():
            serializable_results[target] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in result.items()
            }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存到: {results_file}")


class RegressionExperiment(BaseExperiment):
    """
    回归实验类
    
    支持多种评估指标：RMSE, MAE, MAPE, MedAE, R², Explained Variance
    """
    
    SUPPORTED_METRICS = ['rmse', 'mae', 'mape', 'medae', 'r2', 'explained_variance']
    
    def __init__(self, config: Dict[str, Any], model_type: str = 'lightgbm', target_col: str = None):
        """
        初始化回归实验
        
        Args:
            config: 实验配置字典
            model_type: 模型类型 ('lightgbm', 'tabtransformer', etc.)
            target_col: 目标变量列名
        """
        super().__init__(config)
        self.model_type = model_type
        self.target_col = target_col or config.get('target_col')
        self.metrics = config.get('evaluation', {}).get('metrics', ['rmse', 'mae', 'r2'])
        
        # 初始化组件
        feature_config = config.get('features', {})
        self.feature_builder = FeatureBuilder(
            feature_columns=feature_config.get('feature_columns'),
            categorical_columns=feature_config.get('categorical_columns'),
            fill_missing=feature_config.get('fill_missing', True),
            scale_features=feature_config.get('scale_features', True)
        )
        
        training_config = config.get('training', {})
        self.trainer = CrossValidationTrainer(
            task_type='regression',
            n_folds=training_config.get('n_folds', 5),
            random_state=training_config.get('random_state', 42)
        )
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估指标字典
        """
        results = {}
        
        # RMSE
        if 'rmse' in self.metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAE
        if 'mae' in self.metrics:
            results['mae'] = mean_absolute_error(y_true, y_pred)
        
        # MAPE
        if 'mape' in self.metrics:
            mask = y_true != 0
            if mask.any():
                results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                results['mape'] = np.nan
        
        # MedAE
        if 'medae' in self.metrics:
            results['medae'] = median_absolute_error(y_true, y_pred)
        
        # R²
        if 'r2' in self.metrics:
            results['r2'] = r2_score(y_true, y_pred)
        
        # Explained Variance
        if 'explained_variance' in self.metrics:
            results['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        return results
    
    def _process_data_with_staffinfo(self, save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        使用 StaffInfo 处理数据
        
        Args:
            save_path: 保存路径
            
        Returns:
            处理后的DataFrame
        """
        logger.info(f"加载数据文件: {self.data_file}")
        
        # 读取原始数据
        if str(self.data_file).endswith('.xlsx'):
            raw_data = pd.read_excel(self.data_file)
        else:
            raw_data = pd.read_csv(self.data_file, encoding='utf-8-sig')
        
        logger.info(f"原始数据形状: {raw_data.shape}")
        
        # 通过 StaffInfo 构建数据
        records = []
        compare_iso = self.config.get('evaluation', {}).get('compare_with_iso', False)
        
        for idx, row in raw_data.iterrows():
            if idx % 100 == 0:
                logger.debug(f"处理进度: {idx}/{len(raw_data)}")
            
            try:
                # 准备StaffInfo所需的数据
                row_dict = row.to_dict()
                
                # 字段映射：worker_id -> staff_id
                staff_id = row_dict.get('staff_id') or row_dict.get('worker_id')
                if not staff_id:
                    staff_id = f"worker_{idx}"
                
                # 字段映射：recorder_time -> creation_date
                if 'creation_date' not in row_dict or pd.isna(row_dict.get('creation_date')):
                    # 尝试从 recorder_time 获取
                    if 'recorder_time' in row_dict and not pd.isna(row_dict.get('recorder_time')):
                        row_dict['creation_date'] = row_dict['recorder_time']
                    else:
                        # 如果都没有，使用默认值
                        row_dict['creation_date'] = datetime(2021, 1, 1)
                
                # 构建 StaffInfo 对象（staff_id作为直接参数）
                staff = StaffInfo(
                    staff_id=staff_id,
                    basic_info_dict=row_dict,
                    health_info_dict=row_dict,
                    occhaz_info_dict=row_dict
                )
                
                # 基础信息
                record = {
                    'staff_id': staff_id,  # 统一使用 staff_id
                    'sex': staff.basic_info.sex if hasattr(staff, 'basic_info') else row.get('sex'),
                    'age': staff.basic_info.age if hasattr(staff, 'basic_info') else row.get('age'),
                    'duration': row.get('duration', row.get('接噪工龄')),
                    'LAeq': row.get('LAeq', row.get('Leq', row.get('LEX_8h_LEX_40h_median')))
                }
                
                # 计算 NIHL（使用重命名后的方法）
                try:
                    ear_data = {}
                    for freq in [500, 1000, 2000, 3000, 4000, 6000]:
                        left_key = f'left_ear_{freq}'
                        right_key = f'right_ear_{freq}'
                        if left_key in row and right_key in row:
                            ear_data[left_key] = row[left_key]
                            ear_data[right_key] = row[right_key]
                    
                    if ear_data:
                        nihl_1234 = AuditoryDiagnose.calculate_NIHL(
                            ear_data=ear_data,
                            freq_key="1234",
                            age=record['age'],
                            sex=record['sex'],
                            apply_correction=False
                        )
                        record['NIHL_1234'] = nihl_1234 if not np.isnan(nihl_1234) else None
                        
                        nihl_346 = AuditoryDiagnose.calculate_NIHL(
                            ear_data=ear_data,
                            freq_key="346",
                            age=record['age'],
                            sex=record['sex'],
                            apply_correction=False
                        )
                        record['NIHL_346'] = nihl_346 if not np.isnan(nihl_346) else None
                    else:
                        record['NIHL_1234'] = None
                        record['NIHL_346'] = None
                except Exception as e:
                    logger.debug(f"计算NIHL失败: {e}")
                    record['NIHL_1234'] = None
                    record['NIHL_346'] = None
                
                # 计算 NIPTS（使用 ISO 1999:2023 预测方法）
                try:
                    if all(k in record and record[k] is not None for k in ['LAeq', 'age', 'sex', 'duration']):
                        # NIPTS_1234 (1000, 2000, 3000, 4000 Hz)
                        nipts_1234 = AuditoryDiagnose.predict_NIPTS_iso1999_2023(
                            LAeq=record['LAeq'],
                            age=int(record['age']),
                            sex=record['sex'],
                            duration=float(record['duration']),
                            mean_key=[1000, 2000, 3000, 4000],
                            percentrage=50,
                            NH_limit=True
                        )
                        record['NIPTS_1234'] = nipts_1234 if not np.isnan(nipts_1234) else None
                        
                        # NIPTS_346 (3000, 4000, 6000 Hz)
                        nipts_346 = AuditoryDiagnose.predict_NIPTS_iso1999_2023(
                            LAeq=record['LAeq'],
                            age=int(record['age']),
                            sex=record['sex'],
                            duration=float(record['duration']),
                            mean_key=[3000, 4000, 6000],
                            percentrage=50,
                            NH_limit=True
                        )
                        record['NIPTS_346'] = nipts_346 if not np.isnan(nipts_346) else None
                    else:
                        record['NIPTS_1234'] = None
                        record['NIPTS_346'] = None
                except Exception as e:
                    logger.debug(f"计算NIPTS失败: {e}")
                    record['NIPTS_1234'] = None
                    record['NIPTS_346'] = None
                
                # ISO 1999 对比
                if compare_iso:
                    try:
                        if all(k in record and record[k] is not None for k in ['LAeq', 'age', 'sex', 'duration']):
                            # ISO 2013 - NIPTS_1234
                            nipts_2013_1234 = AuditoryDiagnose.predict_NIPTS_iso1999_2013(
                                LAeq=record['LAeq'],
                                age=int(record['age']),
                                sex=record['sex'],
                                duration=float(record['duration']),
                                mean_key=[1000, 2000, 3000, 4000],
                                percentrage=50,
                                NH_limit=True
                            )
                            record['NIPTS_ISO2013_1234'] = nipts_2013_1234 if not np.isnan(nipts_2013_1234) else None
                            
                            # ISO 2013 - NIPTS_346
                            nipts_2013_346 = AuditoryDiagnose.predict_NIPTS_iso1999_2013(
                                LAeq=record['LAeq'],
                                age=int(record['age']),
                                sex=record['sex'],
                                duration=float(record['duration']),
                                mean_key=[3000, 4000, 6000],
                                percentrage=50,
                                NH_limit=True
                            )
                            record['NIPTS_ISO2013_346'] = nipts_2013_346 if not np.isnan(nipts_2013_346) else None
                            
                            # ISO 2023已经在上面计算过了，直接复制
                            record['NIPTS_ISO2023_1234'] = record['NIPTS_1234']
                            record['NIPTS_ISO2023_346'] = record['NIPTS_346']
                        else:
                            record['NIPTS_ISO2013_1234'] = None
                            record['NIPTS_ISO2013_346'] = None
                            record['NIPTS_ISO2023_1234'] = None
                            record['NIPTS_ISO2023_346'] = None
                    except Exception as e:
                        logger.debug(f"ISO 1999对比失败: {e}")
                        record['NIPTS_ISO2013_1234'] = None
                        record['NIPTS_ISO2013_346'] = None
                        record['NIPTS_ISO2023_1234'] = None
                        record['NIPTS_ISO2023_346'] = None
                
                records.append(record)
                
            except Exception as e:
                logger.debug(f"处理行 {idx} 失败: {e}")
                continue
        
        df = pd.DataFrame(records)
        
        logger.info(f"处理后数据形状（删除空行前）: {df.shape}")
        logger.info(f"生成的列: {df.columns.tolist()}")
        
        # 删除目标变量全为空的行
        target_cols = self.config.get('targets', ['NIHL_1234', 'NIHL_346', 'NIPTS_1234', 'NIPTS_346'])
        # 只删除存在的目标列全为空的行
        existing_targets = [col for col in target_cols if col in df.columns]
        if existing_targets:
            df = df.dropna(subset=existing_targets, how='all')
            logger.info(f"删除目标列全空行后数据形状: {df.shape}")
        else:
            logger.warning(f"警告：没有找到任何目标列 {target_cols}")
        
        logger.info(f"处理后数据形状: {df.shape}")
        logger.info(f"各目标变量有效样本数:")
        for col in target_cols:
            if col in df.columns:
                valid_count = df[col].notna().sum()
                logger.info(f"  {col}: {valid_count}")
            else:
                logger.warning(f"  {col}: 列不存在")
        
        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存到: {save_path}")
        
        return df
    
    def run_data_pipeline(self) -> pd.DataFrame:
        """
        运行数据处理管道
        
        Returns:
            处理后的DataFrame
        """
        logger.info("=" * 60)
        logger.info("步骤1: 数据处理")
        logger.info("=" * 60)
        
        processed_file = self.output_dir / "processed_data.csv"
        use_staffinfo = self.config.get('data', {}).get('use_staffinfo', True)
        
        if processed_file.exists():
            logger.info("加载已处理的数据...")
            self.data = pd.read_csv(processed_file, encoding='utf-8-sig')
        else:
            if use_staffinfo:
                logger.info("使用StaffInfo构建数据...")
                self.data = self._process_data_with_staffinfo(save_path=processed_file)
            else:
                logger.info("直接加载数据...")
                if str(self.data_file).endswith('.xlsx'):
                    self.data = pd.read_excel(self.data_file)
                else:
                    self.data = pd.read_csv(self.data_file, encoding='utf-8-sig')
                self.data.to_csv(processed_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"数据形状: {self.data.shape}")
        return self.data
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征
        
        Returns:
            (特征矩阵X, 目标变量y)
        """
        logger.info("=" * 60)
        logger.info("步骤2: 特征工程")
        logger.info("=" * 60)
        
        X, y = self.feature_builder.prepare_features(self.data, self.target_col, fit=True)
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            
        Returns:
            训练结果
        """
        logger.info("=" * 60)
        logger.info("步骤3: 模型训练")
        logger.info("=" * 60)
        
        # 分割数据
        test_size = self.config.get('training', {}).get('test_size', 0.2)
        X_train, X_test, y_train, y_test = self.trainer.split_data(X, y, test_size=test_size)
        
        # 获取模型参数
        model_params = {}
        for model_config in self.config.get('models', []):
            if model_config['type'] == self.model_type:
                model_params = model_config.get('params', {})
                break
        
        # 交叉验证
        cv_results = self.trainer.cross_validate(
            ModelWrapper, X_train, y_train,
            model_params={'model_type': self.model_type, **model_params}
        )
        
        # 测试集评估
        y_pred = cv_results['best_model'].predict(X_test)
        test_results = self.evaluate(y_test.values, y_pred)
        test_results.update({
            'test_mse': mean_squared_error(y_test, y_pred),
            'y_true': y_test.values.tolist(),
            'y_pred': y_pred.tolist()
        })
        
        # 保存模型
        if self.config.get('output', {}).get('save_models', True):
            model_path = self.output_dir / "models" / f"{self.target_col}_{self.model_type}.pkl"
            self.trainer.save_model(cv_results['best_model'], model_path)
        else:
            model_path = None
        
        # 保存特征重要性
        try:
            importance_df = cv_results['best_model'].get_feature_importance()
            importance_path = self.output_dir / "models" / f"{self.target_col}_{self.model_type}_importance.csv"
            importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.warning(f"无法保存特征重要性: {e}")
        
        results = {
            'model_type': self.model_type,
            'target_col': self.target_col,
            'cv_results': cv_results,
            'test_results': test_results,
            'model_path': str(model_path) if model_path else None
        }
        
        logger.info(f"训练完成: Test RMSE={test_results.get('rmse', 0):.4f}")
        
        self.results = {f"{self.model_type}_{self.target_col}": results}
        return results
    
    def generate_report(self) -> str:
        """
        生成实验报告
        
        Returns:
            报告文件路径
        """
        logger.info("=" * 60)
        logger.info("步骤4: 生成报告")
        logger.info("=" * 60)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            f"# {self.experiment_name} 实验报告",
            "",
            f"**生成时间**: {timestamp}",
            f"**数据文件**: {self.data_file}",
            f"**模型类型**: {self.model_type}",
            f"**目标变量**: {self.target_col}",
            "",
            "---",
            "",
            "## 1. 实验概述",
            "",
            f"- 数据样本数: {len(self.data) if self.data is not None else 'N/A'}",
            f"- 特征数量: {len(self.feature_builder.feature_columns)}",
            "",
            "## 2. 模型性能",
            ""
        ]
        
        # 构建指标表头
        metric_headers = []
        for metric in self.metrics:
            metric_headers.extend([f"CV {metric.upper()}", f"Test {metric.upper()}"])
        
        header_line = "| 模型 | 目标变量 | " + " | ".join(metric_headers) + " |"
        separator = "|" + "|".join(["---"] * (2 + len(metric_headers))) + "|"
        
        report_lines.extend([header_line, separator])
        
        # 填充结果
        for exp_name, result in self.results.items():
            row_values = [result['model_type'], result['target_col']]
            
            for metric in self.metrics:
                cv_key = f'cv_mean_{metric}'
                test_key = metric
                
                cv_val = result['cv_results'].get(cv_key, 0)
                test_val = result['test_results'].get(test_key, 0)
                
                row_values.append(f"{cv_val:.4f}")
                row_values.append(f"{test_val:.4f}")
            
            report_lines.append("| " + " | ".join(row_values) + " |")
        
        report_lines.extend([
            "",
            "## 3. 实验配置",
            "",
            "### 3.1 特征列",
            ""
        ])
        
        for feature in self.feature_builder.feature_columns:
            report_lines.append(f"- {feature}")
        
        report_lines.extend([
            "",
            "### 3.2 训练配置",
            "",
            f"- 交叉验证折数: {self.trainer.n_folds}",
            f"- 随机种子: {self.trainer.random_state}",
            f"- 评估指标: {', '.join(self.metrics)}",
            "",
            "## 4. 输出文件",
            "",
            f"- 处理后的数据: `{self.output_dir / 'processed_data.csv'}`",
            f"- 训练好的模型: `{self.output_dir / 'models/'}`",
            f"- 训练日志: `{self.output_dir / f'{self.experiment_name}.log'}`",
            ""
        ])
        
        # 保存报告
        report_path = self.output_dir / f"{self.experiment_name}_{self.model_type}_{self.target_col}_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"报告已保存到: {report_path}")
        
        return str(report_path)
    
    def run(self) -> Dict:
        """
        运行完整实验流程
        
        Returns:
            实验结果
        """
        logger.info("=" * 60)
        logger.info(f"开始实验: {self.experiment_name}")
        logger.info(f"模型: {self.model_type}, 目标: {self.target_col}")
        logger.info("=" * 60)
        
        # 步骤1: 数据处理
        self.run_data_pipeline()
        
        # 步骤2: 特征工程
        X, y = self.prepare_features()
        
        # 步骤3: 模型训练
        results = self.train_model(X, y)
        
        # 步骤4: 生成报告
        if self.config.get('output', {}).get('generate_report', True):
            self.generate_report()
        
        # 步骤5: 保存结果
        self.save_results()
        
        logger.info("=" * 60)
        logger.info("实验完成!")
        logger.info("=" * 60)
        
        return results


# 为向后兼容保留别名
BaseExperimentOld = RegressionExperiment