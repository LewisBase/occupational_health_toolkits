"""
OHTK NIHL 计算器模块

提供统一的 NIHL（噪声性听力损失）计算接口，支持多种计算方法：
- 标准计算（无年龄校正）
- 年龄校正计算
- 全频率计算
"""

from ohtk.diagnose_info.nihl_predictor.base import (
    BaseNIHLCalculator,
    NIHLCalculationResult
)
from ohtk.diagnose_info.nihl_predictor.factory import (
    NIHLCalculatorFactory,
    get_calculator
)

# 导入具体实现以触发注册
from ohtk.diagnose_info.nihl_predictor import calculators

__all__ = [
    'BaseNIHLCalculator',
    'NIHLCalculationResult',
    'NIHLCalculatorFactory',
    'get_calculator',
]
