# -*- coding: utf-8 -*-
"""
测试前N行数据处理
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ohtk.experiments import RegressionExperiment, ExperimentConfig

def test_first_n_rows(n=10):
    """测试前N行"""
    config_file = project_root / "examples" / "chinese_worker_experiment" / "config.yaml"
    config = ExperimentConfig.from_yaml(config_file)
    
    exp_config = config.config.copy()
    exp_config['target_col'] = 'NIHL_1234'
    
    exp = RegressionExperiment(
        config=exp_config,
        model_type='lightgbm',
        target_col='NIHL_1234'
    )
    
    print("=" * 60)
    print(f"测试前 {n} 行数据处理")
    print("=" * 60)
    
    # 读取原始数据的前N行
    data_file = project_root / exp_config['data']['file_path']
    raw_data = pd.read_excel(data_file)
    raw_data_head = raw_data.head(n)
    
    print(f"\n原始数据列: {raw_data.columns.tolist()[:10]}")
    print(f"原始数据形状: {raw_data.shape}")
    
    # 直接处理前N行
    from datetime import datetime
    from ohtk.staff_info import StaffInfo
    from ohtk.diagnose_info.auditory_diagnose import AuditoryDiagnose
    import numpy as np
    
    records = []
    for idx, row in raw_data_head.iterrows():
        try:
            row_dict = row.to_dict()
            staff_id = row_dict.get('staff_id') or row_dict.get('worker_id') or f"worker_{idx}"
            
            if 'creation_date' not in row_dict or pd.isna(row_dict.get('creation_date')):
                row_dict['creation_date'] = datetime(2021, 1, 1)
            
            staff = StaffInfo(
                staff_id=staff_id,
                basic_info_dict=row_dict,
                health_info_dict=row_dict,
                occhaz_info_dict=row_dict
            )
            
            record = {
                'sex': row_dict.get('sex'),
                'age': row_dict.get('age'),
                'duration': row_dict.get('duration'),
                'LAeq': row_dict.get('LAeq') or row_dict.get('Leq')
            }
            
            # 计算NIHL
            ear_data = {}
            for freq in [1000, 2000, 3000, 4000, 6000]:
                left_key = f'left_ear_{freq}'
                right_key = f'right_ear_{freq}'
                if left_key in row_dict and right_key in row_dict:
                    ear_data[left_key] = row_dict[left_key]
                    ear_data[right_key] = row_dict[right_key]
            
            if ear_data:
                nihl_346 = AuditoryDiagnose.calculate_NIHL(ear_data=ear_data, freq_key="346")
                record['NIHL_346'] = nihl_346 if not np.isnan(nihl_346) else None
            else:
                record['NIHL_346'] = None
            
            # 计算NIPTS
            if all(k in record and record[k] is not None for k in ['LAeq', 'age', 'sex', 'duration']):
                nipts_346 = AuditoryDiagnose.predict_NIPTS_iso1999_2023(
                    LAeq=record['LAeq'],
                    age=int(record['age']),
                    sex=record['sex'],
                    duration=float(record['duration']),
                    mean_key=[3000, 4000, 6000],
                    percentrage=50
                )
                record['NIPTS_346'] = nipts_346 if not np.isnan(nipts_346) else None
            else:
                record['NIPTS_346'] = None
            
            records.append(record)
            
        except Exception as e:
            print(f"处理行 {idx} 失败: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    print(f"\n处理结果:")
    print(f"成功处理行数: {len(df)}")
    print(f"生成的列: {df.columns.tolist()}")
    print(f"\n前5行:")
    print(df.head())
    
    print(f"\n各列统计:")
    print(df.describe())

if __name__ == "__main__":
    test_first_n_rows(100)  # 测试前100行
