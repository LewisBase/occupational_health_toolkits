"""
机器学习 NIPTS 预测器

支持 sklearn 和其他传统机器学习模型
"""
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from loguru import logger

from ohtk.diagnose_info.nipts_predictor.base import (
    BaseNIPTSPredictor,
    NIPTSPredictionResult
)
from ohtk.diagnose_info.nipts_predictor.factory import NIPTSPredictorFactory


class PickleModelPredictor(BaseNIPTSPredictor):
    """通用 Pickle 模型预测器
    
    支持加载 pickle 格式保存的 sklearn 或其他模型
    """
    
    name = "ml_pickle"
    version = "1.0.0"
    supported_ranges = {
        "LAeq": (70, 130),
        "age": (18, 80),
        "duration": (1, 50),
    }
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """初始化预测器
        
        Args:
            model_path: 模型文件路径
            **kwargs: 其他配置参数
        """
        super().__init__(**kwargs)
        self._model = None
        self._model_path = model_path
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """加载模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            with open(model_path, 'rb') as f:
                self._model = pickle.load(f)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
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
        """使用 ML 模型预测 NIPTS
        
        Args:
            LAeq: 等效连续A计权声压级 (dB)
            age: 年龄
            sex: 性别
            duration: 接噪工龄 (年)
            percentrage: 百分位数（部分模型支持）
            mean_key: 频率列表（部分模型支持）
            **kwargs: 其他参数
            
        Returns:
            NIPTSPredictionResult: 预测结果
        """
        if self._model is None:
            return NIPTSPredictionResult(
                value=float('nan'),
                method=self.name,
                metadata={"error": "Model not loaded"}
            )
        
        try:
            import pandas as pd
            
            # 编码性别
            sex_encoded = 1 if self._normalize_sex(sex) == "Male" else 0
            
            # 构建特征
            feature = [[sex_encoded, age, duration, LAeq]]
            feature_df = pd.DataFrame(
                feature,
                columns=["sex_encoder", "age", "duration", "LAeq"]
            )
            
            # 预测
            prediction = self._model.predict(feature_df)[0]
            
            return NIPTSPredictionResult(
                value=float(prediction),
                method=self.name,
                metadata={
                    "model_path": self._model_path,
                    "features": feature[0],
                }
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return NIPTSPredictionResult(
                value=float('nan'),
                method=self.name,
                metadata={"error": str(e)}
            )


class LinearRegressionPredictor(BaseNIPTSPredictor):
    """线性回归 NIPTS 预测器
    
    基于简单线性回归模型预测
    """
    
    name = "ml_linear"
    version = "1.0.0"
    supported_ranges = {
        "LAeq": (70, 130),
        "age": (18, 80),
        "duration": (1, 50),
    }
    
    def __init__(
        self,
        coefficients: Optional[List[float]] = None,
        intercept: float = 0.0,
        **kwargs
    ):
        """初始化预测器
        
        Args:
            coefficients: 回归系数 [sex, age, duration, LAeq]
            intercept: 截距
            **kwargs: 其他配置参数
        """
        super().__init__(**kwargs)
        # 默认系数（示例值，应从训练中获取）
        self._coefficients = coefficients or [0.5, 0.1, 0.3, 0.8]
        self._intercept = intercept
    
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
        """使用线性回归预测 NIPTS
        
        Args:
            LAeq: 等效连续A计权声压级 (dB)
            age: 年龄
            sex: 性别
            duration: 接噪工龄 (年)
            **kwargs: 其他参数
            
        Returns:
            NIPTSPredictionResult: 预测结果
        """
        try:
            # 编码性别
            sex_encoded = 1 if self._normalize_sex(sex) == "Male" else 0
            
            # 特征向量
            features = np.array([sex_encoded, age, duration, LAeq])
            
            # 线性预测
            prediction = np.dot(features, self._coefficients) + self._intercept
            
            return NIPTSPredictionResult(
                value=float(prediction),
                method=self.name,
                metadata={
                    "coefficients": self._coefficients,
                    "intercept": self._intercept,
                }
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return NIPTSPredictionResult(
                value=float('nan'),
                method=self.name,
                metadata={"error": str(e)}
            )


# 注册预测器
NIPTSPredictorFactory.register("ml_pickle", PickleModelPredictor)
NIPTSPredictorFactory.register("ml_linear", LinearRegressionPredictor)
