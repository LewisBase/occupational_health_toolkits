#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全面测试 NIPTS_predict_iso1999_2023 函数
"""

from ohtk.staff_info.staff_info import StaffInfo


def test_function_comprehensive():
    """全面测试函数功能"""
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
    
    print("创建StaffInfo实例成功")
    
    # 测试不同参数组合
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
            }
        },
        {
            "name": "高百分位数测试",
            "params": {
                "percentrage": 75,
                "mean_key": [3000, 4000, 6000],
                "LAeq": 85,
                "age": 30,
                "sex": "M",
                "duration": 10
            }
        },
        {
            "name": "不同频率测试",
            "params": {
                "percentrage": 50,
                "mean_key": [2000, 3000, 4000],
                "LAeq": 85,
                "age": 30,
                "sex": "M",
                "duration": 10
            }
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
            }
        }
    ]
    
    for case in test_cases:
        try:
            result = staff_info.NIPTS_predict_iso1999_2023(**case["params"])
            print(f"{case['name']}: 成功 - 结果: {result}")
        except Exception as e:
            print(f"{case['name']}: 错误 - {e}")
    
    # 测试边界条件
    print("\n测试边界条件:")
    boundary_tests = [
        {
            "name": "低年龄",
            "params": {
                "percentrage": 50,
                "mean_key": [3000, 4000, 6000],
                "LAeq": 85,
                "age": 15,  # 小于20
                "sex": "M",
                "duration": 5
            }
        },
        {
            "name": "高LAeq值",
            "params": {
                "percentrage": 50,
                "mean_key": [3000, 4000, 6000],
                "LAeq": 105,  # 大于100
                "age": 30,
                "sex": "M",
                "duration": 10,
                "extrapolation": "Linear"  # 使用线性外推
            }
        }
    ]
    
    for case in boundary_tests:
        try:
            result = staff_info.NIPTS_predict_iso1999_2023(**case["params"])
            print(f"{case['name']}: 成功 - 结果: {result}")
        except Exception as e:
            print(f"{case['name']}: 错误 - {e}")
    
    # 测试异常情况
    print("\n测试异常情况:")
    try:
        # 不提供必要参数应该引发异常
        result = staff_info.NIPTS_predict_iso1999_2023()
        print("缺少参数测试: 意外成功 - 结果: {result}")
    except ValueError as e:
        print(f"缺少参数测试: 预期错误 - {e}")
    except Exception as e:
        print(f"缺少参数测试: 其他错误 - {e}")


if __name__ == "__main__":
    test_function_comprehensive()