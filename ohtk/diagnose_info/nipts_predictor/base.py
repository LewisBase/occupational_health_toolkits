"""
NIPTS 预测器基类和接口定义

提供统一的 NIPTS 预测接口，支持多种预测方法
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


class NIPTSPredictionResult(BaseModel):
    """NIPTS 预测结果对象"""
    
    value: float  # 预测值 (dB)
    method: str  # 使用的预测方法名称
    confidence: Optional[float] = None  # 置信度（ML/DL 模型可用）
    metadata: Dict[str, Any] = {}  # 额外元数据
    
    model_config = {"arbitrary_types_allowed": True}


class BaseNIPTSPredictor(ABC):
    """NIPTS 预测器抽象基类
    
    所有 NIPTS 预测器必须继承此类并实现 predict 方法
    """
    
    # 预测器名称
    name: str = "base"
    # 版本信息
    version: str = "1.0.0"
    # 支持的参数范围
    supported_ranges: Dict[str, tuple] = {
        "LAeq": (70, 120),
        "age": (18, 70),
        "duration": (1, 40),
    }
    
    def __init__(self, **kwargs):
        """初始化预测器
        
        Args:
            **kwargs: 预测器特定配置参数
        """
        self._config = kwargs
    
    @abstractmethod
    def predict(
        self,
        LAeq: float,
        age: int,
        sex: str,
        duration: float,
        percentrage: int = 50,
        mean_key: Optional[List[int]] = None,
        **kwargs
    ) -> NIPTSPredictionResult:
        """执行 NIPTS 预测
        
        Args:
            LAeq: 等效连续A计权声压级 (dB)
            age: 年龄
            sex: 性别 ('M'/'F' 或 'Male'/'Female')
            duration: 接噪工龄 (年)
            percentrage: 百分位数，默认 50
            mean_key: 频率列表，默认 [3000, 4000, 6000]
            **kwargs: 其他方法特定参数
            
        Returns:
            NIPTSPredictionResult: 预测结果对象
        """
        pass
    
    def batch_predict(
        self,
        data_list: List[Dict],
        **kwargs
    ) -> List[NIPTSPredictionResult]:
        """批量预测
        
        Args:
            data_list: 包含预测参数的字典列表
            **kwargs: 通用预测参数
            
        Returns:
            List[NIPTSPredictionResult]: 预测结果列表
        """
        results = []
        for data in data_list:
            merged_kwargs = {**kwargs, **data}
            try:
                result = self.predict(**merged_kwargs)
            except Exception as e:
                result = NIPTSPredictionResult(
                    value=float('nan'),
                    method=self.name,
                    metadata={"error": str(e)}
                )
            results.append(result)
        return results
    
    def _normalize_sex(self, sex: str) -> str:
        """标准化性别参数
        
        Args:
            sex: 原始性别字符串
            
        Returns:
            标准化后的性别 ('Male' 或 'Female')
        """
        if str(sex).upper().startswith('M') or sex in ('男', 'male'):
            return 'Male'
        return 'Female'
    
    def _validate_inputs(
        self,
        LAeq: float,
        age: int,
        duration: float
    ) -> bool:
        """验证输入参数是否在支持范围内
        
        Args:
            LAeq: 声压级
            age: 年龄
            duration: 工龄
            
        Returns:
            bool: 是否有效
        """
        laeq_range = self.supported_ranges.get("LAeq", (70, 120))
        age_range = self.supported_ranges.get("age", (18, 70))
        duration_range = self.supported_ranges.get("duration", (1, 40))
        
        if not (laeq_range[0] <= LAeq <= laeq_range[1]):
            return False
        if not (age_range[0] <= age <= age_range[1]):
            return False
        if not (duration_range[0] <= duration <= duration_range[1]):
            return False
        return True
    
    def get_info(self) -> Dict:
        """获取预测器信息
        
        Returns:
            Dict: 预测器信息字典
        """
        return {
            "name": self.name,
            "version": self.version,
            "supported_ranges": self.supported_ranges,
            "config": self._config
        }
