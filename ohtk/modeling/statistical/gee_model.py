# -*- coding: utf-8 -*-
"""
广义估计方程（GEE）模型

用于分析纵向/重复测量数据中的暴露-反应关系，
考虑个体内观测的相关性结构。
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from loguru import logger
from typing import List, Optional, Dict, Any


class GEEModel:
    """
    广义估计方程模型
    
    适用于队列研究中重复测量数据的分析，
    可以自动选择最优的相关结构。
    
    Attributes:
        outcome: 结果变量名
        exposure: 暴露变量名
        covariates: 控制变量列表
        group_var: 分组变量名（如个体ID）
    """
    
    # 可用的相关结构
    CORR_STRUCTURES = {
        "Independence": sm.cov_struct.Independence,
        "Exchangeable": sm.cov_struct.Exchangeable,
        "Autoregressive": sm.cov_struct.Autoregressive,
    }
    
    def __init__(
        self,
        outcome: str = "NIHL346",
        exposure: str = "LAeq",
        covariates: Optional[List[str]] = None,
        group_var: str = "worker_id"
    ):
        """
        初始化GEE模型
        
        Args:
            outcome: 结果变量名
            exposure: 暴露变量名
            covariates: 控制变量列表
            group_var: 分组变量名
        """
        self.outcome = outcome
        self.exposure = exposure
        self.covariates = covariates or ["check_order", "sex", "age"]
        self.group_var = group_var
        self.best_result = None
        self.best_structure = None
    
    def fit(
        self,
        df: pd.DataFrame,
        family: str = "gaussian",
        auto_select_corr: bool = True,
        corr_structure: str = "Exchangeable",
        time_var: str = "check_order"
    ) -> Dict[str, Any]:
        """
        拟合GEE模型
        
        Args:
            df: 数据框
            family: 分布族 ("gaussian", "binomial", "poisson")
            auto_select_corr: 是否自动选择最优相关结构
            corr_structure: 指定的相关结构（当auto_select_corr=False时使用）
            time_var: 时间变量名（用于排序，对Autoregressive结构重要）
        
        Returns:
            包含模型结果的字典
        """
        # 数据预处理
        df = df.copy()
        
        # 按组和时间排序（对于Autoregressive结构很重要）
        if time_var in df.columns:
            df = df.sort_values([self.group_var, time_var])
        
        # 确保分组变量是分类类型
        df[self.group_var] = df[self.group_var].astype("category")
        
        # 选择分布族
        family_obj = self._get_family(family)
        
        # 构建公式
        formula = self._build_formula()
        
        if auto_select_corr:
            result = self._auto_select_correlation(df, family_obj, formula)
        else:
            result = self._fit_single(df, family_obj, formula, corr_structure)
        
        return result
    
    def _get_family(self, family: str) -> sm.families.Family:
        """获取分布族对象"""
        families = {
            "gaussian": sm.families.Gaussian(),
            "binomial": sm.families.Binomial(),
            "poisson": sm.families.Poisson(),
        }
        if family not in families:
            raise ValueError(f"不支持的分布族: {family}")
        return families[family]
    
    def _build_formula(self) -> str:
        """构建回归公式"""
        covar_str = " + ".join(self.covariates)
        return f"{self.outcome} ~ {self.exposure} + {covar_str}"
    
    def _fit_single(
        self,
        df: pd.DataFrame,
        family: sm.families.Family,
        formula: str,
        corr_name: str
    ) -> Dict[str, Any]:
        """拟合单个相关结构的模型"""
        corr_struct = self.CORR_STRUCTURES[corr_name]()
        
        model = smf.gee(
            formula,
            self.group_var,
            df,
            family=family,
            cov_struct=corr_struct
        )
        
        result = model.fit()
        
        # qic 在新版本 statsmodels 中可能是方法而不是属性
        qic_raw = result.qic() if callable(result.qic) else result.qic
        qic_value = qic_raw[0] if isinstance(qic_raw, tuple) else qic_raw
        
        # 确保数值类型正确（避免格式化错误）
        exposure_coef = float(result.params[self.exposure])
        exposure_se = float(result.bse[self.exposure])
        exposure_p = float(result.pvalues[self.exposure])
        
        return {
            "model": result,
            "corr_structure": corr_name,
            "qic": float(qic_value),
            "params": result.params.to_dict(),
            "pvalues": result.pvalues.to_dict(),
            "exposure_coef": exposure_coef,
            "exposure_se": exposure_se,
            "exposure_p": exposure_p,
        }
    
    def _auto_select_correlation(
        self,
        df: pd.DataFrame,
        family: sm.families.Family,
        formula: str
    ) -> Dict[str, Any]:
        """自动选择最优相关结构"""
        best_qic = float("inf")
        best_result = None
        all_results = {}
        
        # 优先尝试 Independence 和 Exchangeable（更稳定）
        # Autoregressive 需要数据满足特定条件，放在最后尝试
        corr_order = ["Independence", "Exchangeable", "Autoregressive"]
        
        for name in corr_order:
            if name not in self.CORR_STRUCTURES:
                continue
            try:
                result = self._fit_single(df, family, formula, name)
                all_results[name] = result
                
                logger.debug(f"{name}: QIC = {result['qic']:.2f}")
                
                if result["qic"] < best_qic:
                    best_qic = result["qic"]
                    best_result = result
                    self.best_structure = name
            except Exception as e:
                logger.warning(f"相关结构 {name} 拟合失败: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("所有相关结构都拟合失败")
        
        self.best_result = best_result
        
        logger.info(f"最优相关结构: {self.best_structure}, QIC = {best_qic:.2f}")
        logger.info(f"暴露效应: {best_result['exposure_coef']:.4f} "
                   f"(p = {best_result['exposure_p']:.4f})")
        
        return {
            "best_result": best_result,
            "best_structure": self.best_structure,
            "all_results": all_results,
        }
    
    def summary(self) -> str:
        """获取模型摘要"""
        if self.best_result is None:
            return "模型尚未拟合"
        return str(self.best_result["model"].summary())


def fit_gee(
    df: pd.DataFrame,
    outcome: str = "NIHL346",
    exposure: str = "LAeq",
    covariates: Optional[List[str]] = None,
    group_var: str = "worker_id",
    **kwargs
) -> Dict[str, Any]:
    """
    GEE拟合的便捷函数
    
    Args:
        df: 数据框
        outcome: 结果变量名
        exposure: 暴露变量名
        covariates: 控制变量列表
        group_var: 分组变量名
        **kwargs: 传递给GEEModel.fit()的其他参数
    
    Returns:
        模型结果字典
    """
    model = GEEModel(outcome, exposure, covariates, group_var)
    return model.fit(df, **kwargs)
