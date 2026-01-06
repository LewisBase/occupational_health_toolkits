#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证重构后的 NIPTS_predict_iso1999_2023 函数是否正确
"""

import numpy as np
from ohtk.staff_info.staff_info import StaffInfo


def test_function_validation():
    """验证重构后的函数功能"""
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
            {"LAeq": 85},
            {"LAeq": 85},
            {"LAeq": 85}
        ]
    }

    staff_info = StaffInfo(staff_id="001", **test_data)
    
    print("验证重构后的 NIPTS_predict_iso1999_2023 函数")
    print("=" * 50)
    
    # 测试用例
    test_cases = [
        {
            "name": "标准测试",
            "params": {
                "percentrage": 50,
                "mean_key": [3000, 4000, 6000],
                "LAeq": 85,
                "age": 30,
                "sex": "M",
                "duration": 10
            },
            "expected_type": (int, float, np.floating),
            "description": "标准参数下的计算"
        },
        {
            "name": "女性测试",
            "params": {
                "percentrage": 50,
                "mean_key": [3000, 4000, 6000],
                "LAeq": 85,
                "age": 35,
                "sex": "F",
                "duration": 15
            },
            "expected_type": (int, float, np.floating),
            "description": "女性参数下的计算"
        },
        {
            "name": "边界年龄测试",
            "params": {
                "percentrage": 50,
                "mean_key": [3000, 4000, 6000],
                "LAeq": 85,
                "age": 15,  # 小于20
                "sex": "M",
                "duration": 5
            },
            "expected_type": (int, float, np.floating),
            "description": "年龄小于20的边界条件"
        },
        {
            "name": "外推测试",
            "params": {
                "percentrage": 50,
                "mean_key": [3000, 4000, 6000],
                "LAeq": 105,  # 大于100
                "age": 30,
                "sex": "M",
                "duration": 10,
                "extrapolation": "Linear"
            },
            "expected_type": (int, float, np.floating),
            "description": "LAeq大于100的外推计算"
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(test_cases, 1):
        print(f"测试 {i}: {case['name']}")
        print(f"  描述: {case['description']}")
        print(f"  参数: {case['params']}")
        
        try:
            result = staff_info.NIPTS_predict_iso1999_2023(**case["params"])
            
            # 检查返回值类型
            is_type_correct = isinstance(result, case["expected_type"])
            print(f"  结果: {result}")
            print(f"  类型检查: {'✓' if is_type_correct else '✗'} ({type(result)})")
            
            # 检查是否为NaN
            is_not_nan = not np.isnan(result)
            print(f"  NaN检查: {'✓' if is_not_nan else '✗'}")
            
            if is_type_correct and is_not_nan:
                print(f"  状态: ✓ 通过")
            else:
                print(f"  状态: ✗ 失败")
                all_passed = False
                
        except Exception as e:
            print(f"  错误: {e}")
            print(f"  状态: ✗ 异常")
            all_passed = False
        
        print()
    
    # 测试异常情况
    print("异常情况测试:")
    print("测试缺少必要参数:")
    try:
        # 创建一个没有噪声危害信息的StaffInfo实例，这样_get_staff_data_for_niots将返回None
        test_data_missing = {
            "record_year": [2020],
            "name": ["张三"],
            "factory_name": ["工厂A"],
            "work_shop": ["车间1"],
            "work_position": ["岗位1"],
            "sex": ["M"],
            "age": [30],
            "duration": [10],
            "smoking": [0],
            "year_of_smoking": [0],
            "cigaretee_per_day": [0],
            "occupational_clinic_class": ["A"],
            "auditory_detection": [{}],
            "auditory_diagnose": [{}],
            "noise_hazard_info": [None]  # 没有噪声危害信息
        }

        staff_missing = StaffInfo(staff_id="002", **test_data_missing)
        
        # 应该抛出ValueError，因为缺少LAeq数据
        result = staff_missing.NIPTS_predict_iso1999_2023(
            percentrage=50,
            mean_key=[3000, 4000, 6000]
        )
        print("  状态: ✗ 应该抛出异常但没有")
        all_passed = False
    except ValueError as e:
        if "Required data" in str(e):
            print(f"  状态: ✓ 正确抛出ValueError: {e}")
        else:
            print(f"  状态: ✗ 异常类型正确但消息不匹配: {e}")
            all_passed = False
    except Exception as e:
        print(f"  状态: ✗ 抛出了意外异常: {e}")
        all_passed = False
    
    print()
    print("=" * 50)
    if all_passed:
        print("✓ 所有测试通过！重构后的函数功能正确。")
    else:
        print("✗ 部分测试失败。")
    
    return all_passed


if __name__ == "__main__":
    test_function_validation()