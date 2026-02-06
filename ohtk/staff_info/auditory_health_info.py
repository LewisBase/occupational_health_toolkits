# -*- coding: utf-8 -*-
"""
听力健康检查信息模块

AuditoryHealthInfo 封装所有听力检查相关的字段和方法。
注意：不存储 sex/age，计算方法通过参数获取（从 StaffInfo 传入）。
"""

from pydantic import BaseModel, model_validator
from typing import Dict, Optional, List, Union
import numpy as np


class AuditoryHealthInfo(BaseModel):
    """
    听力健康检查信息
    
    封装所有听力检查相关的字段和方法：
    - detection: 原始检测数据 (PTA, ABR等)
    - diagnose: 诊断结果
    - NIHL: 噪声性听力损失指标
    - 听力损失分类
    
    注意：不存储 sex/age，计算方法通过参数获取
    """
    # 检测数据
    detection: Optional[Dict] = None
    diagnose: Optional[Dict] = None
    
    # NIHL 指标 (值可能为 None，在 validator 中过滤)
    NIHL: Optional[Dict[str, Optional[float]]] = None  # {"1234": 25.5, "346": 32.0}
    norm_hearing_loss: Optional[str] = None   # 基于 NIHL1234 的分类
    high_hearing_loss: Optional[str] = None   # 基于 NIHL346 的分类
    
    # 原始耳数据（用于计算）
    ear_data: Optional[Dict[str, float]] = None
    
    # 内部处理后的检测对象（不参与序列化）
    _detection_objects: Dict = {}
    
    model_config = {"arbitrary_types_allowed": True}
    
    @model_validator(mode='after')
    def process_auditory(self) -> 'AuditoryHealthInfo':
        """处理听力检查数据"""
        if self.diagnose is None:
            self.diagnose = {}
        
        # 过滤 NIHL 字典中的 None 值
        if self.NIHL:
            self.NIHL = {k: v for k, v in self.NIHL.items() if v is not None}
            if not self.NIHL:
                self.NIHL = None
        
        # 处理检测数据
        if self.detection:
            self._process_detection()
        
        # 如果直接提供了 NIHL 值，进行分类
        if self.NIHL:
            self._classify_hearing_loss()
        
        return self
    
    def _process_detection(self):
        """处理听力检测数据，构建检测对象"""
        from ohtk.constants.global_constants import AuditoryNametoObject
        
        for key, value in self.detection.items():
            if key in AuditoryNametoObject.DETECTION_TYPE_DICT:
                self._detection_objects[key] = AuditoryNametoObject.DETECTION_TYPE_DICT[key](data=value)
    
    def calculate_NIHL(
        self,
        sex: str,  # 必须通过参数传入（从 StaffInfo 获取）
        age: int,  # 必须通过参数传入（从 StaffInfo 获取）
        freq_keys: List[str] = None,
        apply_correction: bool = False
    ) -> Dict[str, float]:
        """
        计算 NIHL（调用 AuditoryDiagnose）
        
        Args:
            sex: 性别（从 StaffInfo 传入）
            age: 年龄（从 StaffInfo 传入）
            freq_keys: 频率配置列表，默认 ["1234", "346"]
            apply_correction: 是否应用年龄校正
        
        Returns:
            NIHL 值字典，如 {"1234": 25.5, "346": 32.0}
        
        Raises:
            ValueError: 如果没有可用的耳数据
        """
        if freq_keys is None:
            freq_keys = ["1234", "346"]
        
        from ohtk.diagnose_info import AuditoryDiagnose
        
        if not self.ear_data:
            if self._detection_objects:
                self.ear_data = self._extract_ear_data_from_detection()
            else:
                raise ValueError("No ear data available for NIHL calculation")
        
        result = {}
        for freq_key in freq_keys:
            value = AuditoryDiagnose.NIHL(
                ear_data=self.ear_data,
                freq_key=freq_key,
                age=age,
                sex=sex,
                apply_correction=apply_correction
            )
            if not np.isnan(value):
                result[freq_key] = value
        
        self.NIHL = result
        self._classify_hearing_loss()
        return result
    
    def _extract_ear_data_from_detection(self) -> Dict[str, float]:
        """从检测对象中提取耳数据"""
        ear_data = {}
        
        for detection_key, detection_obj in self._detection_objects.items():
            # 尝试获取左耳数据
            if hasattr(detection_obj, 'left_ear_data') and detection_obj.left_ear_data:
                for freq, value in detection_obj.left_ear_data.items():
                    ear_data[f"left_ear_{freq}"] = value
            # 尝试获取右耳数据
            if hasattr(detection_obj, 'right_ear_data') and detection_obj.right_ear_data:
                for freq, value in detection_obj.right_ear_data.items():
                    ear_data[f"right_ear_{freq}"] = value
        
        return ear_data
    
    def _classify_hearing_loss(self):
        """分类听力损失"""
        from ohtk.utils.pta_correction import classify_hearing_loss
        
        if self.NIHL:
            if "1234" in self.NIHL and self.norm_hearing_loss is None:
                self.norm_hearing_loss = classify_hearing_loss(self.NIHL["1234"])
            if "346" in self.NIHL and self.high_hearing_loss is None:
                self.high_hearing_loss = classify_hearing_loss(self.NIHL["346"])
    
    def get_nihl(self, freq_key: str = "346") -> Optional[float]:
        """
        获取 NIHL 值
        
        Args:
            freq_key: "1234" 或 "346"
        
        Returns:
            NIHL 值，如果不存在返回 None
        """
        return self.NIHL.get(freq_key) if self.NIHL else None
