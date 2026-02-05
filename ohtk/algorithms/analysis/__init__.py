# -*- coding: utf-8 -*-
"""
Analysis 子模块

提供暴露-反应关系分析工具：
- threshold_analysis: 阈值发现
- nonlinear_analysis: 非线性分析
"""

from ohtk.algorithms.analysis.threshold_analysis import (
    PiecewiseThresholdFinder,
    DecisionTreeThresholdFinder,
    find_optimal_threshold,
)
from ohtk.algorithms.analysis.nonlinear_analysis import (
    GAMAnalyzer,
    QuantileRegressionAnalyzer,
    analyze_nonlinearity,
)

__all__ = [
    'PiecewiseThresholdFinder',
    'DecisionTreeThresholdFinder',
    'find_optimal_threshold',
    'GAMAnalyzer',
    'QuantileRegressionAnalyzer',
    'analyze_nonlinearity',
]
