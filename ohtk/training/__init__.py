"""
OHTK Training 模块

包含模型训练相关工具
"""

from ohtk.training.train_model import StepRunner, EpochRunner, train_model
from ohtk.training.multi_task_trainer import train_model as train_multi_task_model

__all__ = [
    'StepRunner',
    'EpochRunner',
    'train_model',
    'train_multi_task_model',
]
