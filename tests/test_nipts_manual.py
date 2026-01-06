import unittest
import numpy as np
from ohtk.staff_info.staff_info import StaffInfo


class TestNIPTSPredictISO19992023Manual(unittest.TestCase):
    """手动测试 NIPTS_predict_iso1999_2023 函数"""

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

    def test_function_exists(self):
        """测试函数是否存在"""
        # 验证函数是否存在
        self.assertTrue(hasattr(self.staff_info, 'NIPTS_predict_iso1999_2023'))
        print("Function exists: NIPTS_predict_iso1999_2023")

    def test_function_signature(self):
        """测试函数签名"""
        import inspect
        sig = inspect.signature(self.staff_info.NIPTS_predict_iso1999_2023)
        params = list(sig.parameters.keys())
        # 检查参数
        self.assertIn('percentrage', params)
        self.assertIn('mean_key', params)
        print(f"Function signature: {sig}")


if __name__ == '__main__':
    unittest.main()