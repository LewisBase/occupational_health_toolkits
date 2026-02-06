"""
OHTK Diagnose Info 模块

包含诊断相关的类和方法
"""

from ohtk.diagnose_info.auditory_diagnose import AuditoryDiagnose

# NIPTS 预测器
from ohtk.diagnose_info.nipts_predictor import (
    BaseNIPTSPredictor,
    NIPTSPredictionResult,
    NIPTSPredictorFactory,
    get_predictor as get_nipts_predictor,
)

# NIHL 计算器
from ohtk.diagnose_info.nihl_predictor import (
    BaseNIHLCalculator,
    NIHLCalculationResult,
    NIHLCalculatorFactory,
    get_calculator as get_nihl_calculator,
)

__all__ = [
    # 诊断类
    'AuditoryDiagnose',
    # NIPTS 预测器
    'BaseNIPTSPredictor',
    'NIPTSPredictionResult',
    'NIPTSPredictorFactory',
    'get_nipts_predictor',
    # NIHL 计算器
    'BaseNIHLCalculator',
    'NIHLCalculationResult',
    'NIHLCalculatorFactory',
    'get_nihl_calculator',
]