"""
NIHL 计算器基类和接口定义

提供统一的 NIHL（噪声性听力损失）计算接口
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


class NIHLCalculationResult(BaseModel):
    """NIHL 计算结果对象"""
    
    value: float  # 计算值 (dB)
    freq_key: str  # 频率配置键
    method: str  # 使用的计算方法
    corrected: bool = False  # 是否应用了年龄校正
    metadata: Dict[str, Any] = {}  # 额外元数据
    
    model_config = {"arbitrary_types_allowed": True}


class BaseNIHLCalculator(ABC):
    """NIHL 计算器抽象基类
    
    所有 NIHL 计算器必须继承此类并实现 calculate 方法
    """
    
    # 计算器名称
    name: str = "base"
    # 版本信息
    version: str = "1.0.0"
    # 支持的频率配置
    supported_freq_keys: List[str] = ["346", "1234", "512"]
    
    def __init__(self, **kwargs):
        """初始化计算器
        
        Args:
            **kwargs: 计算器特定配置参数
        """
        self._config = kwargs
    
    @abstractmethod
    def calculate(
        self,
        ear_data: Dict[str, float],
        freq_key: str = "346",
        age: Optional[int] = None,
        sex: Optional[str] = None,
        apply_correction: bool = False,
        **kwargs
    ) -> NIHLCalculationResult:
        """计算 NIHL 值
        
        Args:
            ear_data: 双耳听阈数据，格式如 {'left_ear_3000': 25.0, 'right_ear_4000': 30.0, ...}
            freq_key: 频率配置键
                - "1234": 1000, 2000, 3000, 4000 Hz (言语频率)
                - "346": 3000, 4000, 6000 Hz (高频)
                - "512": 500, 1000, 2000 Hz (低频)
            age: 年龄（用于年龄校正时必需）
            sex: 性别 'M' 或 'F'（用于年龄校正时必需）
            apply_correction: 是否应用年龄校正
            **kwargs: 其他方法特定参数
            
        Returns:
            NIHLCalculationResult: 计算结果对象
        """
        pass
    
    def calculate_all(
        self,
        ear_data: Dict[str, float],
        age: Optional[int] = None,
        sex: Optional[str] = None,
        apply_correction: bool = False,
        **kwargs
    ) -> Dict[str, NIHLCalculationResult]:
        """计算所有频率配置的 NIHL 值
        
        Args:
            ear_data: 双耳听阈数据
            age: 年龄
            sex: 性别
            apply_correction: 是否应用年龄校正
            **kwargs: 其他参数
            
        Returns:
            Dict[str, NIHLCalculationResult]: 各频率配置的计算结果
        """
        results = {}
        for freq_key in ["346", "1234"]:
            result = self.calculate(
                ear_data=ear_data,
                freq_key=freq_key,
                age=age,
                sex=sex,
                apply_correction=apply_correction,
                **kwargs
            )
            results[freq_key] = result
        return results
    
    def _normalize_sex(self, sex: str) -> str:
        """标准化性别参数
        
        Args:
            sex: 原始性别字符串
            
        Returns:
            标准化后的性别 ('M' 或 'F')
        """
        if str(sex).upper().startswith('M') or sex in ('男', 'male', 'Male'):
            return 'M'
        return 'F'
    
    def get_info(self) -> Dict:
        """获取计算器信息
        
        Returns:
            Dict: 计算器信息字典
        """
        return {
            "name": self.name,
            "version": self.version,
            "supported_freq_keys": self.supported_freq_keys,
            "config": self._config
        }
