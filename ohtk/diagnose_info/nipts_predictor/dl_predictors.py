"""
深度学习 NIPTS 预测器

支持 PyTorch 深度学习模型
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from ohtk.diagnose_info.nipts_predictor.base import (
    BaseNIPTSPredictor,
    NIPTSPredictionResult
)
from ohtk.diagnose_info.nipts_predictor.factory import NIPTSPredictorFactory


class TorchModelPredictor(BaseNIPTSPredictor):
    """通用 PyTorch 模型预测器
    
    支持加载 PyTorch 格式的深度学习模型
    """
    
    name = "dl_torch"
    version = "1.0.0"
    supported_ranges = {
        "LAeq": (70, 130),
        "age": (18, 80),
        "duration": (1, 50),
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_class: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ):
        """初始化预测器
        
        Args:
            model_path: 模型权重文件路径
            model_class: 模型类名称
            device: 运行设备 ('cpu' 或 'cuda')
            **kwargs: 其他配置参数
        """
        super().__init__(**kwargs)
        self._model = None
        self._model_path = model_path
        self._model_class = model_class
        self._device = device
        
        if model_path and model_class:
            self._load_model(model_path, model_class)
    
    def _load_model(self, model_path: str, model_class: str):
        """加载模型
        
        Args:
            model_path: 模型权重文件路径
            model_class: 模型类名称
        """
        try:
            import torch
            
            # 动态导入模型类
            if model_class == "MMoELayer":
                from ohtk.modeling.multi_task.mmoe import MMoELayer
                self._model = MMoELayer()
            elif model_class == "FTTransformer":
                from ohtk.modeling.transformers.ft_transformer import FTTransformer
                # FTTransformer 需要更多配置参数
                logger.warning("FTTransformer requires additional configuration")
                return
            else:
                logger.warning(f"Unknown model class: {model_class}")
                return
            
            # 加载权重
            self._model.load_state_dict(torch.load(model_path, map_location=self._device))
            self._model.to(self._device)
            self._model.eval()
            
            logger.info(f"Loaded {model_class} from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
        """使用深度学习模型预测 NIPTS
        
        Args:
            LAeq: 等效连续A计权声压级 (dB)
            age: 年龄
            sex: 性别
            duration: 接噪工龄 (年)
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
            import torch
            
            # 编码性别
            sex_encoded = 1 if self._normalize_sex(sex) == "Male" else 0
            
            # 构建输入张量
            features = torch.tensor(
                [[sex_encoded, age, duration, LAeq]],
                dtype=torch.float32,
                device=self._device
            )
            
            # 预测
            with torch.no_grad():
                output = self._model(features)
                if isinstance(output, list):
                    # 多任务模型
                    prediction = output[0].item()
                else:
                    prediction = output.item()
            
            return NIPTSPredictionResult(
                value=float(prediction),
                method=self.name,
                metadata={
                    "model_class": self._model_class,
                    "device": self._device,
                }
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return NIPTSPredictionResult(
                value=float('nan'),
                method=self.name,
                metadata={"error": str(e)}
            )


class MMoEPredictor(BaseNIPTSPredictor):
    """MMoE 多任务学习预测器"""
    
    name = "dl_mmoe"
    version = "1.0.0"
    supported_ranges = {
        "LAeq": (70, 130),
        "age": (18, 80),
        "duration": (1, 50),
    }
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """初始化 MMoE 预测器
        
        Args:
            model_path: 模型权重文件路径
            **kwargs: 其他配置参数
        """
        super().__init__(**kwargs)
        self._model_path = model_path
        self._model = None
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """加载 MMoE 模型"""
        try:
            import torch
            from ohtk.modeling.multi_task.mmoe import MMoELayer
            
            self._model = MMoELayer(input_size=4, num_experts=3, num_tasks=1)
            self._model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self._model.eval()
            
            logger.info(f"Loaded MMoE model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load MMoE model: {e}")
    
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
        """使用 MMoE 模型预测 NIPTS"""
        if self._model is None:
            # 如果模型未加载，返回占位结果
            return NIPTSPredictionResult(
                value=float('nan'),
                method=self.name,
                metadata={"error": "Model not loaded. Please provide model_path."}
            )
        
        try:
            import torch
            
            sex_encoded = 1 if self._normalize_sex(sex) == "Male" else 0
            features = torch.tensor(
                [[sex_encoded, age, duration, LAeq]],
                dtype=torch.float32
            )
            
            with torch.no_grad():
                outputs = self._model(features)
                prediction = outputs[0].squeeze().item()
            
            return NIPTSPredictionResult(
                value=float(prediction),
                method=self.name,
                metadata={"model": "MMoE"}
            )
        except Exception as e:
            logger.error(f"MMoE prediction failed: {e}")
            return NIPTSPredictionResult(
                value=float('nan'),
                method=self.name,
                metadata={"error": str(e)}
            )


# 注册预测器
NIPTSPredictorFactory.register("dl_torch", TorchModelPredictor)
NIPTSPredictorFactory.register("dl_mmoe", MMoEPredictor)
