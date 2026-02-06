# -*- coding: utf-8 -*-
"""
职业危害因素信息模块

StaffOccHazInfo 存储员工的职业危害暴露信息。
"""

from pydantic import BaseModel, model_validator
from typing import Union, Dict, List, Optional

from ohtk.hazard_info import NoiseHazard


class StaffOccHazInfo(BaseModel):
    """
    职业危害因素信息
    
    存储员工的职业危害暴露信息，包括噪声、粉尘等。
    """
    staff_id: Union[int, str]
    hazard_type: List[str] = []
    noise_hazard_info: Optional[Union[Dict, NoiseHazard]] = None
    occupational_hazard_info: Optional[Dict[str, Union[str, float]]] = None
    
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode='after')
    def process_hazard_info(self) -> 'StaffOccHazInfo':
        """处理危害因素信息"""
        if self.noise_hazard_info is not None:
            # 记录危害类型
            if "noise" not in self.hazard_type:
                self.hazard_type.append("noise")
            
            # 如果是字典，转换为 NoiseHazard 对象
            if isinstance(self.noise_hazard_info, dict):
                self.noise_hazard_info = NoiseHazard(**self.noise_hazard_info)
        
        return self
