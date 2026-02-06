"""
OHTK NIPTS 预测器模块

提供统一的 NIPTS（噪声性永久阈移）预测接口，支持多种预测方法：
- ISO 1999:2013 标准
- ISO 1999:2023 标准
- 机器学习模型
- 深度学习模型
"""

from ohtk.diagnose_info.nipts_predictor.base import (
    BaseNIPTSPredictor,
    NIPTSPredictionResult
)
from ohtk.diagnose_info.nipts_predictor.factory import (
    NIPTSPredictorFactory,
    get_predictor
)

# 导入具体实现以触发注册
from ohtk.diagnose_info.nipts_predictor import iso_predictors
from ohtk.diagnose_info.nipts_predictor import ml_predictors
from ohtk.diagnose_info.nipts_predictor import dl_predictors

__all__ = [
    'BaseNIPTSPredictor',
    'NIPTSPredictionResult',
    'NIPTSPredictorFactory',
    'get_predictor',
]
