# -*- coding: utf-8 -*-
"""
中国工人噪声暴露数据分析实验 - 多模型多目标对比

基于YAML配置，支持：
- 多模型（LightGBM, TabTransformer）
- 多目标（NIHL_1234, NIHL_346, NIPTS_1234, NIPTS_346）
- ISO 1999 对比
- 横向性能对比报告

使用方法:
    python run_experiment_new.py
"""

import sys
import warnings
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

from ohtk.experiments import RegressionExperiment, ExperimentConfig


def generate_comparison_report(all_results: dict, output_dir: Path, config: ExperimentConfig):
    """
    生成模型间横向对比报告
    
    Args:
        all_results: 所有实验结果字典
        output_dir: 输出目录
        config: 实验配置
    """
    logger.info("=" * 60)
    logger.info("生成横向对比报告...")
    logger.info("=" * 60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        f"# {config['experiment_name']} - 多模型对比报告",
        "",
        f"**生成时间**: {timestamp}",
        "",
        "---",
        "",
        "## 1. 实验概述",
        "",
        f"- 模型数量: {len(config['models'])}",
        f"- 目标变量: {len(config['targets'])}",
        f"- 交叉验证折数: {config['training']['n_folds']}",
        f"- 使用StaffInfo: {config['data']['use_staffinfo']}",
        "",
        "## 2. 模型性能对比",
        ""
    ]
    
    # 按目标变量分组
    targets = config['targets']
    metrics = config['evaluation']['metrics']
    
    for target in targets:
        report_lines.extend([
            f"### 2.{targets.index(target)+1} {target}",
            ""
        ])
        
        # 构建表头
        header = "| 模型 | " + " | ".join([f"CV {m.upper()}" for m in metrics]) + " | " + " | ".join([f"Test {m.upper()}" for m in metrics]) + " |"
        separator = "|" + "|".join(["---"] * (1 + len(metrics) * 2)) + "|"
        
        report_lines.extend([header, separator])
        
        # 填充每个模型的结果
        for exp_name, result in all_results.items():
            if target in exp_name:
                model_name = exp_name.split('_')[0]
                row_data = [model_name]
                
                # CV指标
                for metric in metrics:
                    cv_key = f'cv_mean_{metric}'
                    cv_val = result.get('cv_results', {}).get(cv_key, 0)
                    row_data.append(f"{cv_val:.4f}")
                
                # Test指标
                for metric in metrics:
                    test_key = metric
                    test_val = result.get('test_results', {}).get(test_key, 0)
                    row_data.append(f"{test_val:.4f}")
                
                report_lines.append("| " + " | ".join(row_data) + " |")
        
        report_lines.append("")
    
    # ISO 1999 对比（如果可用）
    if config['evaluation'].get('compare_with_iso', False):
        report_lines.extend([
            "## 3. ISO 1999 对比分析",
            "",
            "（ISO 1999预测结果已集成到数据中，可用于模型训练和对比）",
            ""
        ])
    
    # 最佳模型推荐
    report_lines.extend([
        "## 4. 最佳模型推荐",
        ""
    ])
    
    for target in targets:
        best_model = None
        best_rmse = float('inf')
        
        for exp_name, result in all_results.items():
            if target in exp_name:
                test_rmse = result.get('test_results', {}).get('rmse', float('inf'))
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    best_model = exp_name.split('_')[0]
        
        report_lines.append(f"- **{target}**: {best_model} (Test RMSE: {best_rmse:.4f})")
    
    report_lines.extend([
        "",
        "## 5. 配置信息",
        "",
        "### 5.1 模型配置",
        ""
    ])
    
    for model in config['models']:
        report_lines.append(f"**{model['name']}**")
        report_lines.append("```yaml")
        report_lines.append(f"type: {model['type']}")
        for key, value in model.get('params', {}).items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("```")
        report_lines.append("")
    
    report_lines.extend([
        "### 5.2 训练配置",
        "",
        f"- 交叉验证折数: {config['training']['n_folds']}",
        f"- 测试集比例: {config['training']['test_size']}",
        f"- 随机种子: {config['training']['random_state']}",
        "",
        "---",
        "",
        "*本报告由 ohtk.experiments 自动生成*"
    ])
    
    # 保存报告
    report_path = output_dir / f"{config['experiment_name']}_comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"横向对比报告已保存到: {report_path}")
    return report_path


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("开始多模型多目标实验")
    logger.info("=" * 80)
    
    # 加载配置
    config_path = Path(__file__).parent / "config.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    
    logger.info(f"\n配置信息:")
    logger.info(f"  实验名称: {config['experiment_name']}")
    logger.info(f"  模型数量: {len(config['models'])}")
    logger.info(f"  目标变量: {config['targets']}")
    logger.info(f"  数据文件: {config['data']['file_path']}")
    
    # 创建多个实验
    experiments = []
    for model_config in config['models']:
        for target in config['targets']:
            exp_name = f"{model_config['name']}_{target}"
            
            # 创建实验配置副本
            exp_config = config.config.copy()
            exp_config['experiment_name'] = f"{config['experiment_name']}_{exp_name}"
            exp_config['target_col'] = target
            
            exp = RegressionExperiment(
                config=exp_config,
                model_type=model_config['type'],
                target_col=target
            )
            
            experiments.append((exp_name, exp))
    
    logger.info(f"\n共创建 {len(experiments)} 个实验")
    
    # 运行所有实验
    all_results = {}
    for i, (exp_name, exp) in enumerate(experiments, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"实验 {i}/{len(experiments)}: {exp_name}")
        logger.info("=" * 80)
        
        try:
            result = exp.run()
            all_results[exp_name] = result
            
            # 打印简要结果
            for key, value in result.items():
                if 'test_results' in value:
                    test_rmse = value['test_results'].get('rmse', 0)
                    test_r2 = value['test_results'].get('r2', 0)
                    logger.info(f"  Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        except Exception as e:
            logger.error(f"  实验失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成横向对比报告
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_comparison_report(all_results, output_dir, config.config)
    
    # 打印最终总结
    logger.info("\n" + "=" * 80)
    logger.info("实验完成总结")
    logger.info("=" * 80)
    logger.info(f"成功完成: {len(all_results)}/{len(experiments)} 个实验")
    logger.info(f"详细报告: {output_dir / f'{config['experiment_name']}_comparison_report.md'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
