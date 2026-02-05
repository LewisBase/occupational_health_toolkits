"""
OHTK Algorithms 模块

包含算法工具（非深度学习）
- fitting: 数学拟合函数
- mining: 数据挖掘算法
- analysis: 暴露-反应关系分析
"""

from ohtk.algorithms.fitting import LAeqFunction, NILFunction
from ohtk.algorithms.mining import Apriori
from ohtk.algorithms.analysis import (
    PiecewiseThresholdFinder,
    DecisionTreeThresholdFinder,
    find_optimal_threshold,
    GAMAnalyzer,
    QuantileRegressionAnalyzer,
    analyze_nonlinearity,
)

__all__ = [
    # Fitting
    'LAeqFunction',
    'NILFunction',
    # Mining
    'Apriori',
    # Analysis
    'PiecewiseThresholdFinder',
    'DecisionTreeThresholdFinder',
    'find_optimal_threshold',
    'GAMAnalyzer',
    'QuantileRegressionAnalyzer',
    'analyze_nonlinearity',
]
