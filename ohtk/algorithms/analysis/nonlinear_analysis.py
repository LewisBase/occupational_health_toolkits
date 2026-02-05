# -*- coding: utf-8 -*-
"""
非线性分析模块

提供暴露-反应关系的非线性分析方法：
- GAM（广义可加模型）分析
- 分位数回归分析
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from loguru import logger
from typing import List, Optional, Dict, Any, Tuple


class GAMAnalyzer:
    """
    使用广义可加模型（GAM）分析暴露与结果的非线性关系
    
    GAM可以自动拟合暴露变量的非线性效应，同时控制其他协变量。
    通过计算导数可以识别效应变化的拐点。
    
    Attributes:
        outcome: 结果变量名
        exposure: 暴露变量名
        covariates: 控制变量列表
    
    Note:
        需要安装 pygam 包: pip install pygam
    """
    
    def __init__(
        self,
        outcome: str = "NIHL346",
        exposure: str = "LAeq",
        covariates: Optional[List[str]] = None
    ):
        """
        初始化GAM分析器
        
        Args:
            outcome: 结果变量名
            exposure: 暴露变量名
            covariates: 控制变量列表
        """
        self.outcome = outcome
        self.exposure = exposure
        self.covariates = covariates or ["check_order", "sex", "age"]
        self.gam_model = None
    
    def analyze(
        self,
        df: pd.DataFrame,
        n_splines: int = 20,
        lam: float = 0.6,
        grid_points: int = 100
    ) -> Dict[str, Any]:
        """
        执行GAM分析
        
        Args:
            df: 数据框
            n_splines: 样条基函数数量，控制平滑度
            lam: 正则化参数，防止过拟合
            grid_points: 预测网格点数
        
        Returns:
            包含模型、拐点、预测值等的字典
        """
        try:
            from pygam import LinearGAM, s
        except ImportError:
            logger.error("需要安装 pygam: pip install pygam")
            raise ImportError("pygam not installed")
        
        # 准备数据
        feature_cols = self.covariates + [self.exposure]
        X = df[feature_cols].values
        y = df[self.outcome].values
        
        # 构建GAM模型 - 为每个特征添加平滑项
        n_features = len(feature_cols)
        terms = sum([s(i) for i in range(n_features)])
        
        self.gam_model = LinearGAM(
            terms,
            lam=lam,
            n_splines=n_splines
        ).fit(X, y)
        
        # 生成预测网格
        X_grid = np.zeros((grid_points, n_features))
        exposure_idx = len(self.covariates)
        
        for i in range(n_features):
            if i == exposure_idx:
                X_grid[:, i] = np.linspace(
                    df[self.exposure].min(),
                    df[self.exposure].max(),
                    grid_points
                )
            else:
                X_grid[:, i] = np.median(X[:, i])
        
        # 预测并计算导数
        predictions = self.gam_model.predict(X_grid)
        exposure_values = X_grid[:, exposure_idx]
        
        # 数值导数
        derivative = np.gradient(predictions, exposure_values)
        second_derivative = np.gradient(derivative, exposure_values)
        
        # 寻找拐点
        inflection_points = self._find_inflection_points(
            exposure_values, second_derivative
        )
        
        logger.info(f"GAM分析完成，识别到 {len(inflection_points)} 个拐点")
        if inflection_points:
            logger.info(f"拐点位置: {[f'{p:.2f}' for p in inflection_points]}")
        
        return {
            "gam_model": self.gam_model,
            "inflection_points": inflection_points,
            "exposure_range": (df[self.exposure].min(), df[self.exposure].max()),
            "exposure_values": exposure_values,
            "predictions": predictions,
            "derivative": derivative,
            "second_derivative": second_derivative
        }
    
    def _find_inflection_points(
        self,
        exposure_values: np.ndarray,
        second_derivative: np.ndarray
    ) -> List[float]:
        """寻找拐点（二阶导数局部极大值）"""
        inflection_points = []
        for i in range(1, len(second_derivative) - 1):
            if (second_derivative[i] > second_derivative[i-1] and 
                second_derivative[i] > second_derivative[i+1] and
                second_derivative[i] > 0):
                inflection_points.append(exposure_values[i])
        return inflection_points


class QuantileRegressionAnalyzer:
    """
    使用分位数回归分析暴露效应在不同结果分布位置上的变化
    
    分位数回归可以揭示暴露对结果变量不同分位数的影响差异，
    识别高风险人群的特殊效应。
    
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
        初始化分位数回归分析器
        
        Args:
            outcome: 结果变量名
            exposure: 暴露变量名
            covariates: 控制变量列表
        """
        self.outcome = outcome
        self.exposure = exposure
        self.covariates = covariates or ["check_order", "sex", "age"]
    
    def analyze(
        self,
        df: pd.DataFrame,
        quantiles: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        执行分位数回归分析
        
        Args:
            df: 数据框
            quantiles: 要分析的分位数列表
        
        Returns:
            包含各分位数回归结果的字典
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        results_dict = {}
        
        # 构建公式
        covar_str = " + ".join(self.covariates)
        formula = f"{self.outcome} ~ {self.exposure} + {covar_str}"
        
        for q in quantiles:
            try:
                model = smf.quantreg(formula, data=df)
                result = model.fit(q=q)
                
                results_dict[q] = {
                    "model": result,
                    "exposure_coef": result.params[self.exposure],
                    "exposure_ci_lower": result.conf_int().loc[self.exposure, 0],
                    "exposure_ci_upper": result.conf_int().loc[self.exposure, 1],
                    "exposure_p": result.pvalues[self.exposure],
                    "pseudo_r2": result.prsquared
                }
                
                logger.debug(
                    f"Q{q:.2f}: coef={result.params[self.exposure]:.4f}, "
                    f"p={result.pvalues[self.exposure]:.4f}"
                )
            except Exception as e:
                logger.warning(f"分位数 {q} 回归失败: {e}")
                continue
        
        # 创建效应汇总表
        quantile_effects = self._summarize_effects(results_dict)
        
        # 计算效应变异系数
        effects = [r["exposure_coef"] for r in results_dict.values()]
        effect_variation = (
            np.std(effects) / np.mean(np.abs(effects)) if effects else 0
        )
        
        logger.info(
            f"分位数回归分析完成，分析了 {len(results_dict)} 个分位数"
        )
        logger.info(f"效应变异系数: {effect_variation:.4f}")
        
        return {
            "results": results_dict,
            "effect_variation_coefficient": effect_variation,
            "quantile_effects": quantile_effects
        }
    
    def _summarize_effects(
        self, 
        results_dict: Dict[float, Dict[str, Any]]
    ) -> pd.DataFrame:
        """汇总各分位数的效应"""
        data = []
        for q, result in results_dict.items():
            data.append({
                "quantile": q,
                "effect": result["exposure_coef"],
                "ci_lower": result["exposure_ci_lower"],
                "ci_upper": result["exposure_ci_upper"],
                "p_value": result["exposure_p"],
                "pseudo_r2": result["pseudo_r2"]
            })
        return pd.DataFrame(data)


def analyze_nonlinearity(
    df: pd.DataFrame,
    outcome: str = "NIHL346",
    exposure: str = "LAeq",
    covariates: Optional[List[str]] = None,
    method: str = "quantile",
    **kwargs
) -> Dict[str, Any]:
    """
    非线性分析的便捷函数
    
    Args:
        df: 数据框
        outcome: 结果变量名
        exposure: 暴露变量名
        covariates: 控制变量列表
        method: 分析方法 ("gam" 或 "quantile")
        **kwargs: 传递给具体分析器的参数
    
    Returns:
        分析结果字典
    """
    if method == "gam":
        analyzer = GAMAnalyzer(outcome, exposure, covariates)
        return analyzer.analyze(df, **kwargs)
    elif method == "quantile":
        analyzer = QuantileRegressionAnalyzer(outcome, exposure, covariates)
        return analyzer.analyze(df, **kwargs)
    else:
        raise ValueError(f"未知方法: {method}")
