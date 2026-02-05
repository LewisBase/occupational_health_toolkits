# -*- coding: utf-8 -*-
"""
Boosting 子模块

提供梯度提升模型：
- lgbm_model: LightGBM模型
"""

from ohtk.modeling.boosting.lgbm_model import (
    LGBMNIHLPredictor,
    train_lgbm_nihl,
)

__all__ = [
    'LGBMNIHLPredictor',
    'train_lgbm_nihl',
]
