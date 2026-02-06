"""
NIHL 计算器工厂

提供统一的计算器创建接口
"""
from typing import Dict, Optional, Type

from loguru import logger

from ohtk.diagnose_info.nihl_predictor.base import BaseNIHLCalculator


class NIHLCalculatorFactory:
    """NIHL 计算器工厂类
    
    管理计算器的注册和创建
    """
    
    _registry: Dict[str, Type[BaseNIHLCalculator]] = {}
    _instances: Dict[str, BaseNIHLCalculator] = {}
    
    @classmethod
    def register(cls, name: str, calculator_class: Type[BaseNIHLCalculator]):
        """注册计算器类
        
        Args:
            name: 计算器名称
            calculator_class: 计算器类
        """
        cls._registry[name] = calculator_class
        logger.debug(f"Registered calculator: {name}")
    
    @classmethod
    def create(
        cls,
        method: str = "standard",
        cache: bool = True,
        **kwargs
    ) -> BaseNIHLCalculator:
        """创建计算器实例
        
        Args:
            method: 计算方法名称
            cache: 是否缓存实例
            **kwargs: 计算器配置参数
            
        Returns:
            BaseNIHLCalculator: 计算器实例
            
        Raises:
            ValueError: 当指定的方法未注册时
        """
        if method not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown calculator method: '{method}'. "
                f"Available methods: {available}"
            )
        
        # 检查缓存
        cache_key = f"{method}_{hash(frozenset(kwargs.items()))}"
        if cache and cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # 创建新实例
        calculator_class = cls._registry[method]
        instance = calculator_class(**kwargs)
        
        if cache:
            cls._instances[cache_key] = instance
        
        return instance
    
    @classmethod
    def list_methods(cls) -> list:
        """列出所有可用的计算方法
        
        Returns:
            list: 方法名称列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def clear_cache(cls):
        """清除实例缓存"""
        cls._instances.clear()


def get_calculator(method: str = "standard", **kwargs) -> BaseNIHLCalculator:
    """获取计算器的便捷函数
    
    Args:
        method: 计算方法名称
        **kwargs: 计算器配置参数
        
    Returns:
        BaseNIHLCalculator: 计算器实例
    """
    return NIHLCalculatorFactory.create(method, **kwargs)
