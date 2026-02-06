"""
NIHL 计算器实现

提供标准和年龄校正的 NIHL 计算
"""
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from ohtk.diagnose_info.nihl_predictor.base import (
    BaseNIHLCalculator,
    NIHLCalculationResult
)
from ohtk.diagnose_info.nihl_predictor.factory import NIHLCalculatorFactory
from ohtk.utils.pta_correction import (
    calculate_nihl,
    calculate_all_nihl,
    classify_hearing_loss
)


class StandardNIHLCalculator(BaseNIHLCalculator):
    """标准 NIHL 计算器
    
    使用 pta_correction 模块进行计算，不应用年龄校正
    """
    
    name = "standard"
    version = "1.0.0"
    
    def calculate(
        self,
        ear_data: Dict[str, float],
        freq_key: str = "346",
        age: Optional[int] = None,
        sex: Optional[str] = None,
        apply_correction: bool = False,
        **kwargs
    ) -> NIHLCalculationResult:
        """计算 NIHL 值（标准方法，无年龄校正）
        
        Args:
            ear_data: 双耳听阈数据
            freq_key: 频率配置键
            age: 年龄（不使用）
            sex: 性别（不使用）
            apply_correction: 是否应用年龄校正（强制为 False）
            **kwargs: 其他参数
            
        Returns:
            NIHLCalculationResult: 计算结果
        """
        try:
            value = calculate_nihl(
                ear_data=ear_data,
                freq_key=freq_key,
                age=None,
                sex=None,
                apply_correction=False
            )
            
            # 分类听力损失
            classification = None
            if not np.isnan(value):
                classification = classify_hearing_loss(value)
            
            return NIHLCalculationResult(
                value=float(value) if not np.isnan(value) else float('nan'),
                freq_key=freq_key,
                method=self.name,
                corrected=False,
                metadata={
                    "classification": classification,
                }
            )
        except Exception as e:
            logger.error(f"NIHL calculation failed: {e}")
            return NIHLCalculationResult(
                value=float('nan'),
                freq_key=freq_key,
                method=self.name,
                corrected=False,
                metadata={"error": str(e)}
            )


class CorrectedNIHLCalculator(BaseNIHLCalculator):
    """年龄校正 NIHL 计算器
    
    应用年龄-性别校正因子
    """
    
    name = "corrected"
    version = "1.0.0"
    
    def calculate(
        self,
        ear_data: Dict[str, float],
        freq_key: str = "346",
        age: Optional[int] = None,
        sex: Optional[str] = None,
        apply_correction: bool = True,
        **kwargs
    ) -> NIHLCalculationResult:
        """计算 NIHL 值（带年龄校正）
        
        Args:
            ear_data: 双耳听阈数据
            freq_key: 频率配置键
            age: 年龄（必需）
            sex: 性别（必需）
            apply_correction: 是否应用年龄校正（默认 True）
            **kwargs: 其他参数
            
        Returns:
            NIHLCalculationResult: 计算结果
        """
        if age is None or sex is None:
            return NIHLCalculationResult(
                value=float('nan'),
                freq_key=freq_key,
                method=self.name,
                corrected=False,
                metadata={"error": "Age and sex are required for corrected calculation"}
            )
        
        try:
            sex_norm = self._normalize_sex(sex)
            
            value = calculate_nihl(
                ear_data=ear_data,
                freq_key=freq_key,
                age=age,
                sex=sex_norm,
                apply_correction=True
            )
            
            # 分类听力损失
            classification = None
            if not np.isnan(value):
                classification = classify_hearing_loss(value)
            
            return NIHLCalculationResult(
                value=float(value) if not np.isnan(value) else float('nan'),
                freq_key=freq_key,
                method=self.name,
                corrected=True,
                metadata={
                    "age": age,
                    "sex": sex_norm,
                    "classification": classification,
                }
            )
        except Exception as e:
            logger.error(f"Corrected NIHL calculation failed: {e}")
            return NIHLCalculationResult(
                value=float('nan'),
                freq_key=freq_key,
                method=self.name,
                corrected=True,
                metadata={"error": str(e)}
            )


class AllFreqNIHLCalculator(BaseNIHLCalculator):
    """全频率 NIHL 计算器
    
    一次性计算所有频率配置的 NIHL 值
    """
    
    name = "all_freq"
    version = "1.0.0"
    
    def calculate(
        self,
        ear_data: Dict[str, float],
        freq_key: str = "346",
        age: Optional[int] = None,
        sex: Optional[str] = None,
        apply_correction: bool = False,
        **kwargs
    ) -> NIHLCalculationResult:
        """计算指定频率的 NIHL 值
        
        Args:
            ear_data: 双耳听阈数据
            freq_key: 频率配置键
            age: 年龄
            sex: 性别
            apply_correction: 是否应用年龄校正
            **kwargs: 其他参数
            
        Returns:
            NIHLCalculationResult: 计算结果
        """
        try:
            sex_norm = self._normalize_sex(sex) if sex else None
            
            all_values = calculate_all_nihl(
                ear_data=ear_data,
                age=age,
                sex=sex_norm,
                apply_correction=apply_correction
            )
            
            value = all_values.get(freq_key, np.nan)
            
            return NIHLCalculationResult(
                value=float(value) if not np.isnan(value) else float('nan'),
                freq_key=freq_key,
                method=self.name,
                corrected=apply_correction,
                metadata={
                    "all_values": all_values,
                }
            )
        except Exception as e:
            logger.error(f"All-freq NIHL calculation failed: {e}")
            return NIHLCalculationResult(
                value=float('nan'),
                freq_key=freq_key,
                method=self.name,
                corrected=apply_correction,
                metadata={"error": str(e)}
            )


# 注册计算器
NIHLCalculatorFactory.register("standard", StandardNIHLCalculator)
NIHLCalculatorFactory.register("corrected", CorrectedNIHLCalculator)
NIHLCalculatorFactory.register("all_freq", AllFreqNIHLCalculator)
