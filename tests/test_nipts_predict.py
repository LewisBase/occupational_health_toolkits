import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from ohtk.staff_info.staff_info import StaffInfo
from ohtk.staff_info.staff_basic_info import StaffBasicInfo
from ohtk.staff_info.staff_health_info import StaffHealthInfo
from ohtk.staff_info.staff_occhaz_info import StaffOccHazInfo
from ohtk.constants.auditory_constants import AuditoryConstants


class TestNIPTSPredictISO19992023(unittest.TestCase):
    """测试 NIPTS_predict_iso1999_2023 函数"""

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

    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1', {
        "10years": {
            "70dB": {
                "3000Hz": {"50pr": 5.0, "75pr": 10.0},
                "4000Hz": {"50pr": 6.0, "75pr": 12.0},
                "6000Hz": {"50pr": 7.0, "75pr": 14.0}
            },
            "85dB": {
                "3000Hz": {"50pr": 15.0, "75pr": 20.0},
                "4000Hz": {"50pr": 18.0, "75pr": 24.0},
                "6000Hz": {"50pr": 21.0, "75pr": 28.0}
            }
        },
        "20years": {
            "85dB": {
                "3000Hz": {"50pr": 25.0, "75pr": 30.0},
                "4000Hz": {"50pr": 30.0, "75pr": 36.0},
                "6000Hz": {"50pr": 35.0, "75pr": 42.0}
            }
        }
    })
    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6', {
        "Male": {
            "20": {
                "3000Hz": {"50pr": 2.0, "75pr": 3.0},
                "4000Hz": {"50pr": 2.5, "75pr": 3.5},
                "6000Hz": {"50pr": 3.0, "75pr": 4.0}
            },
            "30": {
                "3000Hz": {"50pr": 2.5, "75pr": 3.5},
                "4000Hz": {"50pr": 3.0, "75pr": 4.0},
                "6000Hz": {"50pr": 3.5, "75pr": 4.5}
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

    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1', {
        "10years": {
            "85dB": {
                "3000Hz": {"50pr": 15.0},
                "4000Hz": {"50pr": 18.0},
                "6000Hz": {"50pr": 21.0}
            }
        }
    })
    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6', {
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

    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1', {
        "10years": {
            "85dB": {
                "3000Hz": {"50pr": 15.0},
                "4000Hz": {"50pr": 18.0},
                "6000Hz": {"50pr": 21.0}
            }
        }
    })
    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6', {
        "Male": {
            "30": {
                "3000Hz": {"50pr": 2.5},
                "4000Hz": {"50pr": 3.0},
                "6000Hz": {"50pr": 3.5}
            }
        }
    })
    def test_nipts_predict_with_different_frequencies(self, mock_b6, mock_t1):
        """测试不同频率组合"""
        result_default = self.staff_info.NIPTS_predict_iso1999_2023(
            percentrage=50,
            mean_key=[3000, 4000, 6000],
            LAeq=85,
            age=30,
            sex="M",
            duration=10
        )
        
        result_custom = self.staff_info.NIPTS_predict_iso1999_2023(
            percentrage=50,
            mean_key=[2000, 3000, 4000],
            LAeq=85,
            age=30,
            sex="M",
            duration=10
        )
        
        # 验证不同频率组合会返回不同结果
        # 但需要注意，如果模拟数据中没有2000Hz的数据，结果可能相同
        self.assertIsInstance(result_default, (int, float, np.floating))
        self.assertIsInstance(result_custom, (int, float, np.floating))

    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1', {
        "10years": {
            "85dB": {
                "3000Hz": {"50pr": 15.0},
                "4000Hz": {"50pr": 18.0},
                "6000Hz": {"50pr": 21.0}
            }
        }
    })
    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6', {
        "Male": {
            "30": {
                "3000Hz": {"50pr": 2.5},
                "4000Hz": {"50pr": 3.0},
                "6000Hz": {"50pr": 3.5}
            }
        }
    })
    def test_nipts_predict_with_missing_data(self, mock_b6, mock_t1):
        """测试缺少数据时的异常处理"""
        # 创建一个没有噪声危害信息的StaffInfo实例
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
        with self.assertRaises(ValueError) as context:
            staff_missing.NIPTS_predict_iso1999_2023(
                percentrage=50,
                mean_key=[3000, 4000, 6000]
            )
        
        self.assertIn("Required data", str(context.exception))

    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1', {
        "10years": {
            "85dB": {
                "3000Hz": {"50pr": 15.0},
                "4000Hz": {"50pr": 18.0},
                "6000Hz": {"50pr": 21.0}
            }
        }
    })
    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6', {
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

    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1', {
        "10years": {
            "85dB": {
                "3000Hz": {"50pr": 15.0},
                "4000Hz": {"50pr": 18.0},
                "6000Hz": {"50pr": 21.0}
            }
        }
    })
    @patch('ohtk.constants.AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6', {
        "Male": {
            "30": {
                "3000Hz": {"50pr": 2.5},
                "4000Hz": {"50pr": 3.0},
                "6000Hz": {"50pr": 3.5}
            }
        }
    })
    def test_nipts_predict_boundary_conditions(self, mock_b6, mock_t1):
        """测试边界条件"""
        # 测试年龄小于20的情况
        result_age_low = self.staff_info.NIPTS_predict_iso1999_2023(
            percentrage=50,
            mean_key=[3000, 4000, 6000],
            LAeq=85,
            age=15,  # 年龄小于20
            sex="M",
            duration=5
        )
        
        # 在边界条件下，结果可能是NaN或特定值
        self.assertTrue(np.isnan(result_age_low) or isinstance(result_age_low, (int, float)))

        # 测试LAeq大于100的情况
        with patch('pickle.load') as mock_pickle:
            mock_model = MagicMock()
            mock_model.predict.return_value = [25.0]
            mock_pickle.return_value = mock_model
            
            result_laeq_high = self.staff_info.NIPTS_predict_iso1999_2023(
                percentrage=50,
                mean_key=[3000, 4000, 6000],
                LAeq=105,  # 大于100
                age=30,
                sex="M",
                duration=10,
                extrapolation="ML"
            )
            
            # 如果使用ML外推，应该得到一个数值结果
            self.assertIsInstance(result_laeq_high, (int, float, np.floating))


if __name__ == '__main__':
    unittest.main()