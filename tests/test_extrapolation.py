#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试外推功能
"""

from ohtk.staff_info.staff_info import StaffInfo


def test_extrapolation():
    """测试外推功能"""
    # 创建测试数据
    test_data = {
        "record_year": [2020, 2021, 2022],
        "name": ["张三", "张三", "张三"],
        "factory_name": ["工厂A", "工厂A", "工厂A"],
        "work_shop": ["车间1", "车间1", "车间1"],
        "work_position": ["岗位1", "岗位1", "岗位1"],
        "sex": ["M", "M", "M"],
        "age": [30, 31, 32],
        "duration": [5, 6, 7],
        "smoking": [0, 0, 0],
        "year_of_smoking": [0, 0, 0],
        "cigaretee_per_day": [0, 0, 0],
        "occupational_clinic_class": ["A", "A", "A"],
        "auditory_detection": [{}, {}, {}],  # 修改为字典类型
        "auditory_diagnose": [{}, {}, {}],  # 修改为字典类型
        "noise_hazard_info": [
            {"LAeq": 105},  # 高LAeq值，用于测试外推
            {"LAeq": 105},
            {"LAeq": 105}
        ]
    }

    staff_info = StaffInfo(staff_id="001", **test_data)
    
    print("创建StaffInfo实例成功，LAeq=105用于测试外推")
    
    try:
        result = staff_info.NIPTS_predict_iso1999_2023(
            LAeq=105,  # 高LAeq值
            age=30,
            sex="M",
            duration=10,
            extrapolation="Linear",
            percentrage=50,
            mean_key=[3000, 4000, 6000]
        )
        print(f"外推测试成功 - 结果: {result}")
    except Exception as e:
        print(f"外推测试失败 - 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_extrapolation()