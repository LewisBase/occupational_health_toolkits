# -*- coding: utf-8 -*-
"""
Cox比例风险模型（生存分析）

用于分析暴露因素对听力损失发生时间的影响。
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Optional, Dict, Any


class CoxPHModel:
    """
    Cox比例风险模型
    
    用于分析队列数据中暴露因素对NIHL发生风险的影响。
    
    Attributes:
        outcome: 结果变量名（用于定义事件）
        exposure: 暴露变量名
        covariates: 控制变量列表
        threshold: 定义事件的阈值
        group_var: 个体标识变量
    """
    
    def __init__(
        self,
        outcome: str = "NIHL346",
        exposure: str = "LAeq",
        covariates: Optional[List[str]] = None,
        threshold: float = 25.0,
        group_var: str = "worker_id",
        time_var: str = "check_order"
    ):
        """
        初始化Cox模型
        
        Args:
            outcome: 结果变量名
            exposure: 暴露变量名
            covariates: 控制变量列表
            threshold: 定义事件的NIHL阈值（dB）
            group_var: 个体标识变量
            time_var: 时间变量
        """
        self.outcome = outcome
        self.exposure = exposure
        self.covariates = covariates or ["age", "sex"]
        self.threshold = threshold
        self.group_var = group_var
        self.time_var = time_var
        self.cph = None
        self.survival_df = None
    
    def prepare_survival_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备生存分析数据
        
        将纵向数据转换为生存分析格式，
        定义事件时间和删失状态。
        
        Args:
            df: 原始纵向数据
        
        Returns:
            生存分析格式的数据框
        """
        survival_data = []
        
        for worker_id, group in df.groupby(self.group_var):
            group = group.sort_values(self.time_var)
            
            # 检查是否发生事件（超过阈值）
            event_mask = group[self.outcome] > self.threshold
            
            if event_mask.any():
                # 首次发生事件的时间
                event_time = group.loc[event_mask, self.time_var].iloc[0]
                event_status = 1
            else:
                # 删失：最后一次观察的时间
                event_time = group[self.time_var].iloc[-1]
                event_status = 0
            
            # 基线特征（第一次检查时的特征）
            baseline = group.iloc[0]
            
            record = {
                self.group_var: worker_id,
                "duration": event_time,
                "event": event_status,
                f"baseline_{self.exposure}": baseline.get(self.exposure, np.nan),
                f"baseline_{self.outcome}": baseline.get(self.outcome, np.nan),
            }
            
            # 添加协变量
            for cov in self.covariates:
                record[f"baseline_{cov}"] = baseline.get(cov, np.nan)
            
            survival_data.append(record)
        
        self.survival_df = pd.DataFrame(survival_data)
        
        n_events = self.survival_df["event"].sum()
        n_total = len(self.survival_df)
        
        logger.info(f"生存数据准备完成: {n_total} 个个体, {n_events} 个事件 "
                   f"({n_events/n_total*100:.1f}%)")
        
        return self.survival_df
    
    def fit(
        self,
        df: pd.DataFrame,
        penalizer: float = 0.1
    ) -> Dict[str, Any]:
        """
        拟合Cox模型
        
        Args:
            df: 原始纵向数据（会自动转换为生存数据）
            penalizer: 正则化参数
        
        Returns:
            模型结果字典
        """
        try:
            from lifelines import CoxPHFitter
            from lifelines.utils import concordance_index
        except ImportError:
            logger.error("需要安装 lifelines: pip install lifelines")
            raise ImportError("lifelines not installed")
        
        # 准备生存数据
        if self.survival_df is None:
            self.prepare_survival_data(df)
        
        # 构建特征列
        feature_cols = ["duration", "event"]
        feature_cols.extend([f"baseline_{self.exposure}"])
        feature_cols.extend([f"baseline_{cov}" for cov in self.covariates])
        
        # 过滤有效数据
        analysis_df = self.survival_df[feature_cols].dropna()
        
        # 拟合模型
        self.cph = CoxPHFitter(penalizer=penalizer)
        self.cph.fit(
            analysis_df,
            duration_col="duration",
            event_col="event"
        )
        
        # 计算一致性指数
        c_index = concordance_index(
            analysis_df["duration"],
            -self.cph.predict_partial_hazard(analysis_df),
            analysis_df["event"]
        )
        
        logger.info(f"Cox模型拟合完成，一致性指数 (C-index): {c_index:.4f}")
        
        # 提取结果
        exposure_var = f"baseline_{self.exposure}"
        
        return {
            "model": self.cph,
            "c_index": c_index,
            "summary": self.cph.summary,
            "hazard_ratios": self.cph.hazard_ratios_.to_dict(),
            "exposure_hr": self.cph.hazard_ratios_[exposure_var],
            "exposure_p": self.cph.summary.loc[exposure_var, "p"],
            "n_samples": len(analysis_df),
            "n_events": analysis_df["event"].sum(),
        }
    
    def check_assumptions(self, p_threshold: float = 0.05) -> Dict[str, Any]:
        """
        检查比例风险假设
        
        Args:
            p_threshold: 显著性水平
        
        Returns:
            检验结果
        """
        if self.cph is None:
            raise RuntimeError("请先拟合模型")
        
        # 获取用于检验的数据
        feature_cols = ["duration", "event"]
        feature_cols.extend([f"baseline_{self.exposure}"])
        feature_cols.extend([f"baseline_{cov}" for cov in self.covariates])
        analysis_df = self.survival_df[feature_cols].dropna()
        
        try:
            # 执行Schoenfeld残差检验
            results = self.cph.check_assumptions(
                analysis_df,
                p_value_threshold=p_threshold,
                show_plots=False
            )
            
            return {
                "passed": True,
                "details": results
            }
        except Exception as e:
            logger.warning(f"比例风险假设检验: {e}")
            return {
                "passed": False,
                "message": str(e)
            }
    
    def summary(self) -> str:
        """获取模型摘要"""
        if self.cph is None:
            return "模型尚未拟合"
        return str(self.cph.summary)


def fit_cox(
    df: pd.DataFrame,
    outcome: str = "NIHL346",
    exposure: str = "LAeq",
    covariates: Optional[List[str]] = None,
    threshold: float = 25.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Cox模型拟合的便捷函数
    
    Args:
        df: 纵向数据框
        outcome: 结果变量名
        exposure: 暴露变量名
        covariates: 控制变量列表
        threshold: 定义事件的阈值
        **kwargs: 传递给CoxPHModel.fit()的其他参数
    
    Returns:
        模型结果字典
    """
    model = CoxPHModel(outcome, exposure, covariates, threshold)
    return model.fit(df, **kwargs)
