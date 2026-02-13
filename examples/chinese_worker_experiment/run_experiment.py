# -*- coding: utf-8 -*-
"""
中国工人噪声暴露数据分析实验

单一可执行脚本，直接生成Markdown报告

使用方法:
    cd examples/chinese_worker_experiment
    python run_experiment.py

或在VS Code中使用"Python: 当前文件"配置进行调试
"""

import sys
import warnings
from pathlib import Path
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

from ohtk.experiments import BaseExperiment


def main():
    """主函数"""
    # 配置路径
    data_file = project_root / "examples" / "data" / "All_Chinese_worker_exposure_data_0401.xlsx"
    output_dir = Path(__file__).parent / "results"
    
    # 检查数据文件
    if not data_file.exists():
        logger.info(f"错误: 数据文件不存在: {data_file}")
        logger.info("请确保数据文件位于 examples/data/ 目录下")
        return
    
    # 创建实验
    experiment = BaseExperiment(
        data_file=data_file,
        output_dir=output_dir,
        experiment_name="chinese_worker_nihl_analysis"
    )
    
    # 运行实验
    results = experiment.run(target_cols=["NIHL_1234", "NIHL_346"])
    
    # 打印结果摘要
    logger.info("\n" + "=" * 60)
    logger.info("实验结果摘要")
    logger.info("=" * 60)
    
    for target_name, result in results.items():
        logger.info(f"\n{target_name}:")
        logger.info(f"  CV RMSE: {result['cv_results']['cv_mean_rmse']:.4f} ± {result['cv_results']['cv_std_rmse']:.4f}")
        logger.info(f"  CV R²:   {result['cv_results']['cv_mean_r2']:.4f} ± {result['cv_results']['cv_std_r2']:.4f}")
        logger.info(f"  Test RMSE: {result['test_results']['test_rmse']:.4f}")
        logger.info(f"  Test R²:   {result['test_results']['test_r2']:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"详细报告已保存到: {output_dir / 'chinese_worker_nihl_analysis_report.md'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()