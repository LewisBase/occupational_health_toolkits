# -*- coding: utf-8 -*-
"""
测试数据处理 - 验证NIPTS列生成
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ohtk.experiments import RegressionExperiment, ExperimentConfig

def test_data_processing():
    """测试数据处理是否生成NIPTS列"""
    config_file = project_root / "examples" / "chinese_worker_experiment" / "config.yaml"
    config = ExperimentConfig.from_yaml(config_file)
    
    # 创建一个实验实例
    exp_config = config.config.copy()
    exp_config['target_col'] = 'NIHL_1234'
    
    exp = RegressionExperiment(
        config=exp_config,
        model_type='lightgbm',
        target_col='NIHL_1234'
    )
    
    print("=" * 60)
    print("测试数据处理")
    print("=" * 60)
    
    # 运行数据处理
    df = exp.run_data_pipeline()
    
    print("\n生成的数据列:")
    print(df.columns.tolist())
    
    print(f"\n数据形状: {df.shape}")
    
    print("\n各列非空数量:")
    for col in df.columns:
        count = df[col].notna().sum()
        print(f"  {col}: {count}")
    
    # 检查NIPTS列
    print("\n" + "=" * 60)
    print("NIPTS列检查:")
    print("=" * 60)
    
    target_cols = ['NIHL_1234', 'NIHL_346', 'NIPTS_1234', 'NIPTS_346']
    for col in target_cols:
        if col in df.columns:
            count = df[col].notna().sum()
            print(f"✅ {col}: {count} 个有效样本")
        else:
            print(f"❌ {col}: 列不存在！")
    
    # 显示前几行NIPTS数据
    if 'NIPTS_346' in df.columns:
        print("\n前5行NIPTS数据:")
        print(df[['age', 'sex', 'duration', 'LAeq', 'NIPTS_1234', 'NIPTS_346']].head())

if __name__ == "__main__":
    test_data_processing()
