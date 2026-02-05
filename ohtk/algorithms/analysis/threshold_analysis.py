# -*- coding: utf-8 -*-
"""
阈值分析模块

提供噪声暴露阈值发现的多种方法：
- 分段线性回归
- 决策树递归识别
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from loguru import logger
from typing import List, Optional, Dict, Any, Tuple


class PiecewiseThresholdFinder:
    """
    通过分段线性回归寻找暴露变量的最佳阈值
    
    该方法在不同候选阈值处将暴露变量分为两段，
    分别估计其对结果变量的影响，选择使模型拟合最优的阈值。
    
    Attributes:
        outcome: 结果变量名
        exposure: 暴露变量名
        covariates: 控制变量列表
    """
    
    def __init__(
        self,
        outcome: str = "NIHL346",
        exposure: str = "LAeq",
        covariates: Optional[List[str]] = None
    ):
        """
        初始化分段阈值发现器
        
        Args:
            outcome: 结果变量名（如 NIHL346）
            exposure: 暴露变量名（如 LAeq）
            covariates: 控制变量列表
        """
        self.outcome = outcome
        self.exposure = exposure
        self.covariates = covariates or ["check_order", "sex", "age"]
    
    def find_threshold(
        self,
        df: pd.DataFrame,
        min_percentile: int = 10,
        max_percentile: int = 90,
        step: int = 1
    ) -> Tuple[float, pd.DataFrame]:
        """
        寻找最佳阈值
        
        Args:
            df: 包含变量的DataFrame
            min_percentile: 搜索的最小百分位数
            max_percentile: 搜索的最大百分位数
            step: 搜索步长（百分位数单位）
        
        Returns:
            optimal_threshold: 最佳阈值
            results_df: 所有候选阈值的结果
        """
        results = []
        
        # 生成候选阈值（基于暴露变量的百分位数）
        percentiles = np.arange(min_percentile, max_percentile + step, step)
        candidate_thresholds = np.percentile(df[self.exposure], percentiles)
        
        for threshold in candidate_thresholds:
            result = self._evaluate_threshold(df, threshold)
            if result is not None:
                results.append(result)
        
        if not results:
            logger.warning("所有阈值拟合都失败了")
            return None, pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # 综合评分函数
        results_df["score"] = self._calculate_score(results_df)
        
        optimal_idx = results_df["score"].idxmax()
        optimal_threshold = results_df.loc[optimal_idx, "threshold"]
        
        logger.info(f"找到最佳阈值: {optimal_threshold:.2f}")
        
        return optimal_threshold, results_df
    
    def _evaluate_threshold(
        self, 
        df: pd.DataFrame, 
        threshold: float
    ) -> Optional[Dict[str, Any]]:
        """评估单个阈值"""
        df_temp = df.copy()
        
        # 创建分段变量
        df_temp[f"{self.exposure}_below"] = np.minimum(
            df_temp[self.exposure], threshold
        )
        df_temp[f"{self.exposure}_above"] = np.maximum(
            df_temp[self.exposure] - threshold, 0
        )
        
        # 构建回归公式
        covar_str = " + ".join(self.covariates)
        formula = (
            f"{self.outcome} ~ {covar_str} + "
            f"{self.exposure}_below + {self.exposure}_above"
        )
        
        try:
            model = sm.OLS.from_formula(formula, data=df_temp).fit()
            
            coef_below = model.params[f"{self.exposure}_below"]
            coef_above = model.params[f"{self.exposure}_above"]
            
            return {
                "threshold": threshold,
                "coef_below": coef_below,
                "coef_above": coef_above,
                "coef_diff": coef_above - coef_below,
                "coef_diff_p": model.pvalues[f"{self.exposure}_above"],
                "aic": model.aic,
                "bic": model.bic,
                "r2": model.rsquared,
                "adj_r2": model.rsquared_adj,
                "n_below": (df_temp[self.exposure] <= threshold).sum(),
                "n_above": (df_temp[self.exposure] > threshold).sum()
            }
        except Exception as e:
            logger.debug(f"阈值 {threshold:.1f} 拟合失败: {e}")
            return None
    
    def _calculate_score(self, results_df: pd.DataFrame) -> pd.Series:
        """
        计算综合评分
        
        评分基于：
        1. 系数差异显著性（p值最小）
        2. 模型拟合优度（AIC最小）
        3. 解释方差（调整R²最高）
        """
        p_score = -np.log10(results_df["coef_diff_p"].clip(lower=1e-10)) * 0.4
        
        aic_range = results_df["aic"].max() - results_df["aic"].min()
        if aic_range > 0:
            aic_score = (
                (results_df["aic"].max() - results_df["aic"]) / aic_range * 0.3
            )
        else:
            aic_score = 0.3
        
        r2_score = results_df["adj_r2"] * 0.3
        
        return p_score + aic_score + r2_score


class DecisionTreeThresholdFinder:
    """
    使用决策树递归识别多个潜在阈值
    
    决策树自动学习数据中的分割点，可以识别多个阈值，
    特别适合存在多个非线性变化点的情况。
    
    Attributes:
        outcome: 结果变量名
        exposure: 暴露变量名
        covariates: 控制变量列表
        max_depth: 树的最大深度
    """
    
    def __init__(
        self,
        outcome: str = "NIHL346",
        exposure: str = "LAeq",
        covariates: Optional[List[str]] = None,
        max_depth: int = 3
    ):
        """
        初始化决策树阈值发现器
        
        Args:
            outcome: 结果变量名
            exposure: 暴露变量名
            covariates: 控制变量列表
            max_depth: 树的最大深度
        """
        self.outcome = outcome
        self.exposure = exposure
        self.covariates = covariates or ["check_order", "sex", "age"]
        self.max_depth = max_depth
        self.tree_model = None
    
    def find_thresholds(
        self,
        df: pd.DataFrame,
        cv_folds: int = 5,
        min_samples_split: int = 50,
        min_samples_leaf: int = 25
    ) -> Dict[str, Any]:
        """
        识别阈值
        
        Args:
            df: 数据框
            cv_folds: 交叉验证折数
            min_samples_split: 分裂所需最小样本数
            min_samples_leaf: 叶节点最小样本数
        
        Returns:
            包含阈值、特征重要性等信息的字典
        """
        features = self.covariates + [self.exposure]
        X = df[features].values
        y = df[self.outcome].values
        
        # 构建决策树
        self.tree_model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # 交叉验证评估
        cv_scores = cross_val_score(
            self.tree_model, X, y, 
            cv=cv_folds, 
            scoring="neg_mean_squared_error"
        )
        cv_mse = -cv_scores.mean()
        
        # 在全部数据上拟合
        self.tree_model.fit(X, y)
        
        # 提取阈值
        exposure_thresholds = self._extract_thresholds(features)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            "feature": features,
            "importance": self.tree_model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        logger.info(f"识别到 {len(exposure_thresholds)} 个阈值: {exposure_thresholds}")
        logger.info(f"交叉验证 MSE: {cv_mse:.4f}")
        
        return {
            "thresholds": sorted(exposure_thresholds),
            "feature_importance": feature_importance,
            "cv_mse": cv_mse,
            "tree_model": self.tree_model
        }
    
    def _extract_thresholds(self, features: List[str]) -> List[float]:
        """从决策树中提取暴露变量的分割点"""
        n_nodes = self.tree_model.tree_.node_count
        children_left = self.tree_model.tree_.children_left
        children_right = self.tree_model.tree_.children_right
        feature = self.tree_model.tree_.feature
        threshold = self.tree_model.tree_.threshold
        
        exposure_thresholds = []
        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # 内部节点
                if features[feature[i]] == self.exposure:
                    exposure_thresholds.append(threshold[i])
        
        return exposure_thresholds


def find_optimal_threshold(
    df: pd.DataFrame,
    outcome: str = "NIHL346",
    exposure: str = "LAeq",
    covariates: Optional[List[str]] = None,
    method: str = "piecewise"
) -> Tuple[Any, pd.DataFrame]:
    """
    寻找最佳阈值的便捷函数
    
    Args:
        df: 数据框
        outcome: 结果变量名
        exposure: 暴露变量名
        covariates: 控制变量列表
        method: 方法选择 ("piecewise" 或 "tree")
    
    Returns:
        threshold: 最佳阈值（或阈值列表）
        results: 详细结果
    """
    if method == "piecewise":
        finder = PiecewiseThresholdFinder(outcome, exposure, covariates)
        return finder.find_threshold(df)
    elif method == "tree":
        finder = DecisionTreeThresholdFinder(outcome, exposure, covariates)
        result = finder.find_thresholds(df)
        return result["thresholds"], result["feature_importance"]
    else:
        raise ValueError(f"未知方法: {method}")
