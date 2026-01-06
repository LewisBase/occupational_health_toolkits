import unittest
import numpy as np
from unittest.mock import patch
from ohtk.staff_info.staff_info import StaffInfo


class TestNIPTSPredictISO19992023Simple(unittest.TestCase):
    """简单测试 NIPTS_predict_iso1999_2023 函数"""

    def setUp(self):
        """设置测试数据"""
        # 创建模拟数据
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

        self.staff_info = StaffInfo(staff_id="001", **test_data)

    @patch('ohtk.staff_info.staff_info.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1', {
        "10years": {
            "85dB": {
                "3000Hz": {"50pr": 15.0},
                "4000Hz": {"50pr": 18.0},
                "6000Hz": {"50pr": 21.0}
            }
        }
    })
    @patch('ohtk.staff_info.staff_info.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6', {
        "Male": {
            "30": {
                "3000Hz": {"50pr": 2.5},
                "4000Hz": {"50pr": 3.0},
                "6000Hz": {"50pr": 3.5}
            }
        }
    })
    def test_nipts_predict_basic_functionality(self, mock_b6, mock_t1):
        """测试基本功能"""
        # 使用模拟数据测试基本功能
        result = self.staff_info.NIPTS_predict_iso1999_2023(
            percentrage=50,
            mean_key=[3000, 4000, 6000],
            LAeq=85,
            age=30,
            sex="M",
            duration=10
        )
        
        # 验证返回值是数值类型
        self.assertIsInstance(result, (int, float, np.floating))
        # 验证返回值不是NaN
        self.assertFalse(np.isnan(result))
        # 验证返回值在合理范围内（根据模拟数据）
        self.assertGreaterEqual(result, 0)
        print(f"Basic functionality test result: {result}")

    @patch('ohtk.staff_info.staff_info.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1', {
        "10years": {
            "85dB": {
                "3000Hz": {"50pr": 15.0},
                "4000Hz": {"50pr": 18.0},
                "6000Hz": {"50pr": 21.0}
            }
        }
    })
    @patch('ohtk.staff_info.staff_info.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6', {
        "Male": {
            "30": {
                "3000Hz": {"50pr": 2.5},
                "4000Hz": {"50pr": 3.0},
                "6000Hz": {"50pr": 3.5}
            }
        }
    })
    def test_nipts_predict_with_kwargs(self, mock_b6, mock_t1):
        """测试使用kwargs参数"""
        result = self.staff_info.NIPTS_predict_iso1999_2023(
            percentrage=50,
            mean_key=[3000, 4000, 6000],
            LAeq=90,  # 通过kwargs传递
            NH_limit=True  # 通过kwargs传递
        )
        
        self.assertIsInstance(result, (int, float, np.floating))
        self.assertFalse(np.isnan(result))
        print(f"Kwargs test result: {result}")

    @patch('ohtk.staff_info.staff_info.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1', {
        "10years": {
            "85dB": {
                "3000Hz": {"50pr": 15.0},
                "4000Hz": {"50pr": 18.0},
                "6000Hz": {"50pr": 21.0}
            }
        }
    })
    @patch('ohtk.staff_info.staff_info.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6', {
        "Male": {
            "30": {
                "3000Hz": {"50pr": 2.5},
                "4000Hz": {"50pr": 3.0},
                "6000Hz": {"50pr": 3.5}
            }
        }
    })
    def test_nipts_predict_with_different_percentages(self, mock_b6, mock_t1):
        """测试不同百分位数"""
        result_50 = self.staff_info.NIPTS_predict_iso1999_2023(
            percentrage=50,
            mean_key=[3000, 4000, 6000],
            LAeq=85,
            age=30,
            sex="M",
            duration=10
        )
        
        result_75 = self.staff_info.NIPTS_predict_iso1999_2023(
            percentrage=75,
            mean_key=[3000, 4000, 6000],
            LAeq=85,
            age=30,
            sex="M",
            duration=10
        )
        
        # 验证不同百分位数返回不同结果
        self.assertNotEqual(result_50, result_75)
        print(f"Different percentages test results: 50%={result_50}, 75%={result_75}")


if __name__ == '__main__':
    unittest.main()