# -*- coding: utf-8 -*-
"""
简单测试 - 验证单行数据处理
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ohtk.staff_info import StaffInfo
from ohtk.diagnose_info.auditory_diagnose import AuditoryDiagnose

def test_single_row():
    """测试处理单行数据"""
    # 模拟一行数据
    row_dict = {
        'worker_id': 'TEST001',
        'sex': 'M',
        'age': 35,
        'duration': 10.0,
        'LAeq': 85.5,
        'Leq': 85.5,
        'left_ear_1000': 15.0,
        'right_ear_1000': 20.0,
        'left_ear_2000': 25.0,
        'right_ear_2000': 30.0,
        'left_ear_3000': 35.0,
        'right_ear_3000': 40.0,
        'left_ear_4000': 45.0,
        'right_ear_4000': 50.0,
        'left_ear_6000': 55.0,
        'right_ear_6000': 60.0,
    }
    
    print("=" * 60)
    print("测试单行数据处理")
    print("=" * 60)
    
    try:
        # 创建StaffInfo
        staff = StaffInfo(
            staff_id=row_dict['worker_id'],
            basic_info_dict=row_dict,
            health_info_dict=row_dict,
            occhaz_info_dict=row_dict
        )
        print("✅ StaffInfo 创建成功")
    except Exception as e:
        print(f"❌ StaffInfo 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试NIHL计算
    ear_data = {
        'left_ear_3000': 35.0,
        'right_ear_3000': 40.0,
        'left_ear_4000': 45.0,
        'right_ear_4000': 50.0,
        'left_ear_6000': 55.0,
        'right_ear_6000': 60.0,
    }
    
    try:
        nihl_346 = AuditoryDiagnose.calculate_NIHL(
            ear_data=ear_data,
            freq_key="346",
            age=35,
            sex='M',
            apply_correction=False
        )
        print(f"✅ NIHL_346 计算成功: {nihl_346:.2f} dB")
    except Exception as e:
        print(f"❌ NIHL 计算失败: {e}")
        return False
    
    # 测试NIPTS计算
    try:
        nipts_346 = AuditoryDiagnose.predict_NIPTS_iso1999_2023(
            LAeq=85.5,
            age=35,
            sex='M',
            duration=10.0,
            mean_key=[3000, 4000, 6000],
            percentrage=50
        )
        print(f"✅ NIPTS_346 计算成功: {nipts_346:.2f} dB")
    except Exception as e:
        print(f"❌ NIPTS 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_single_row()
