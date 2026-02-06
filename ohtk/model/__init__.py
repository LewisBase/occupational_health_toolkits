"""
OHTK Model 模块 - 向后兼容层

警告: 此模块已重构，建议使用新路径：
- 模型架构定义: ohtk.modeling
- 训练工具: ohtk.training
- 算法工具: ohtk.algorithms

此目录现在仅用于存放训练好的模型权重文件 (.pkl, .pth, .pt 等)
"""
import warnings

# 发出弃用警告
warnings.warn(
    "从 ohtk.model 导入已弃用，请使用新路径：\n"
    "- ohtk.modeling (模型架构)\n"
    "- ohtk.training (训练工具)\n"
    "- ohtk.algorithms (算法工具)",
    DeprecationWarning,
    stacklevel=2
)

# 重定向模型架构
from ohtk.modeling.conv.cnn import CNNModel
from ohtk.modeling.multi_task.mmoe import MMoELayer, MMoEembedding
from ohtk.modeling.multi_task.esmm import ESMM
from ohtk.modeling.multi_task.cnn_mmoe import ConvMMoEModel
from ohtk.modeling.transformers.ft_transformer import FTTransformer
from ohtk.modeling.transformers.tab_transformer_pytorch import TabTransformer
from ohtk.modeling.custom.linear_regression import (
    SegmentAdjustTestModel,
    CustomLayer,
    CustomLayerMono,
    SegmentAdjustModel
)

# 重定向训练工具
from ohtk.training.train_model import StepRunner, EpochRunner, train_model

# 重定向算法工具
from ohtk.algorithms.fitting.noise_functions import LAeqFunction
from ohtk.algorithms.fitting.hearing_loss_functions import NILFunction
from ohtk.algorithms.mining.association_mining import Apriori

__all__ = [
    # 模型架构
    'CNNModel',
    'MMoELayer',
    'MMoEembedding',
    'ESMM',
    'ConvMMoEModel',
    'FTTransformer',
    'TabTransformer',
    'SegmentAdjustTestModel',
    'CustomLayer',
    'CustomLayerMono',
    'SegmentAdjustModel',
    # 训练工具
    'StepRunner',
    'EpochRunner',
    'train_model',
    # 算法工具
    'LAeqFunction',
    'NILFunction',
    'Apriori',
]
