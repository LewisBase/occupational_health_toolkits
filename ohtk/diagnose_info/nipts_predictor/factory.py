"""
NIPTS 预测器工厂

提供统一的预测器创建接口
"""
from typing import Dict, Optional, Type

from loguru import logger

from ohtk.diagnose_info.nipts_predictor.base import BaseNIPTSPredictor


class NIPTSPredictorFactory:
    """NIPTS 预测器工厂类
    
    管理预测器的注册和创建
    """
    
    _registry: Dict[str, Type[BaseNIPTSPredictor]] = {}
    _instances: Dict[str, BaseNIPTSPredictor] = {}
    
    @classmethod
    def register(cls, name: str, predictor_class: Type[BaseNIPTSPredictor]):
        """注册预测器类
        
        Args:
            name: 预测器名称
            predictor_class: 预测器类
        """
        cls._registry[name] = predictor_class
        logger.debug(f"Registered predictor: {name}")
    
    @classmethod
    def create(
        cls,
        method: str,
        cache: bool = True,
        **kwargs
    ) -> BaseNIPTSPredictor:
        """创建预测器实例
        
        Args:
            method: 预测方法名称
            cache: 是否缓存实例
            **kwargs: 预测器配置参数
            
        Returns:
            BaseNIPTSPredictor: 预测器实例
            
        Raises:
            ValueError: 当指定的方法未注册时
        """
        if method not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown predictor method: '{method}'. "
                f"Available methods: {available}"
            )
        
        # 检查缓存
        cache_key = f"{method}_{hash(frozenset(kwargs.items()))}"
        if cache and cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # 创建新实例
        predictor_class = cls._registry[method]
        instance = predictor_class(**kwargs)
        
        if cache:
            cls._instances[cache_key] = instance
        
        return instance
    
    @classmethod
    def list_methods(cls) -> list:
        """列出所有可用的预测方法
        
        Returns:
            list: 方法名称列表
        """
        return list(cls._registry.keys())
    
    @classmethod
    def clear_cache(cls):
        """清除实例缓存"""
        cls._instances.clear()
    
    @classmethod
    def get_predictor_info(cls, method: str) -> Optional[Dict]:
        """获取预测器信息
        
        Args:
            method: 预测方法名称
            
        Returns:
            Dict: 预测器信息，如果方法不存在则返回 None
        """
        if method not in cls._registry:
            return None
        
        predictor_class = cls._registry[method]
        return {
            "name": predictor_class.name,
            "version": predictor_class.version,
            "supported_ranges": predictor_class.supported_ranges,
        }


# 预定义方法别名
METHOD_ALIASES = {
    "iso_2013": "iso1999_2013",
    "iso_2023": "iso1999_2023",
    "ml": "ml_regression",
    "dl": "dl_mmoe",
}


def get_predictor(method: str, **kwargs) -> BaseNIPTSPredictor:
    """获取预测器的便捷函数
    
    支持方法别名
    
    Args:
        method: 预测方法名称或别名
        **kwargs: 预测器配置参数
        
    Returns:
        BaseNIPTSPredictor: 预测器实例
    """
    # 处理别名
    actual_method = METHOD_ALIASES.get(method, method)
    return NIPTSPredictorFactory.create(actual_method, **kwargs)
