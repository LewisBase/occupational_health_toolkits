"""
OHTK Modeling 模块

包含所有模型架构定义
- conv: CNN模型
- multi_task: 多任务学习模型
- transformers: Transformer模型
- custom: 自定义模型
- statistical: 统计模型（GEE, Cox）
- boosting: 梯度提升模型（LightGBM）
"""

# CNN 模型
from ohtk.modeling.conv.cnn import CNNModel

# 多任务学习模型
from ohtk.modeling.multi_task.mmoe import MMoELayer, MMoEembedding
from ohtk.modeling.multi_task.esmm import ESMM
from ohtk.modeling.multi_task.cnn_mmoe import ConvMMoEModel

# Transformer 模型
from ohtk.modeling.transformers.ft_transformer import FTTransformer
from ohtk.modeling.transformers.tab_transformer_pytorch import TabTransformer

# 自定义模型
from ohtk.modeling.custom.linear_regression import (
    SegmentAdjustTestModel,
    CustomLayer,
    CustomLayerMono,
    SegmentAdjustModel
)

# 统计模型
from ohtk.modeling.statistical import (
    GEEModel,
    fit_gee,
    CoxPHModel,
    fit_cox,
)

# Boosting 模型
from ohtk.modeling.boosting import (
    LGBMNIHLPredictor,
    train_lgbm_nihl,
)

__all__ = [
    # CNN
    'CNNModel',
    # Multi-task
    'MMoELayer',
    'MMoEembedding',
    'ESMM',
    'ConvMMoEModel',
    # Transformers
    'FTTransformer',
    'TabTransformer',
    # Custom
    'SegmentAdjustTestModel',
    'CustomLayer',
    'CustomLayerMono',
    'SegmentAdjustModel',
    # Statistical
    'GEEModel',
    'fit_gee',
    'CoxPHModel',
    'fit_cox',
    # Boosting
    'LGBMNIHLPredictor',
    'train_lgbm_nihl',
]
