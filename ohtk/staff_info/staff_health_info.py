# -*- coding: utf-8 -*-
"""
职业健康检查信息统一入口模块

StaffHealthInfo 是职业健康检查所有项目的统一入口，采用嵌套结构管理各检查项目。
注意：不存储 sex/age，从 StaffInfo 获取。
"""

from pydantic import BaseModel, model_validator
from typing import Dict, List, Optional, Union

from ohtk.staff_info.auditory_health_info import AuditoryHealthInfo


class StaffHealthInfo(BaseModel):
    """
    职业健康检查信息统一入口（嵌套结构）
    
    通过嵌套子类管理各检查项目：
    - auditory: 听力检查 (AuditoryHealthInfo)
    - pulmonary: 肺功能检查（未来扩展）
    - blood: 血液检查（未来扩展）
    
    注意：
    - 不存储 sex/age，从 StaffInfo 获取
    - 各检查项目的计算方法需要显式传入 sex/age 参数
    """
    staff_id: Union[int, str]
    
    # === 通用信息 ===
    diagnose_types: List[str] = []  # 已进行的检查类型
    health_summary: Optional[Dict[str, Union[str, float]]] = None  # 健康摘要
    
    # === 检查项目（嵌套结构） ===
    auditory: Optional[AuditoryHealthInfo] = None  # 听力检查
    # pulmonary: Optional[PulmonaryHealthInfo] = None  # 肺功能检查（未来扩展）
    # blood: Optional[BloodTestInfo] = None            # 血液检查（未来扩展）
    # urine: Optional[UrineTestInfo] = None            # 尿检（未来扩展）
    # ecg: Optional[ECGInfo] = None                    # 心电图（未来扩展）
    
    @model_validator(mode='after')
    def process_health_info(self) -> 'StaffHealthInfo':
        """处理健康检查信息"""
        # 自动记录已进行的检查类型
        if self.auditory is not None:
            if "auditory" not in self.diagnose_types:
                self.diagnose_types.append("auditory")
        
        # 未来扩展：
        # if self.pulmonary is not None:
        #     if "pulmonary" not in self.diagnose_types:
        #         self.diagnose_types.append("pulmonary")
        
        return self
    
    # === 听力检查便捷方法 ===
    
    def get_auditory_nihl(self, freq_key: str = "346") -> Optional[float]:
        """
        获取听力 NIHL 值
        
        Args:
            freq_key: "1234" 或 "346"
        
        Returns:
            NIHL 值，如果不存在返回 None
        """
        if self.auditory:
            return self.auditory.get_nihl(freq_key)
        return None
    
    def has_auditory_data(self) -> bool:
        """检查是否有听力检查数据"""
        return self.auditory is not None
    
    # === 未来扩展便捷方法 ===
    # def get_pulmonary_fvc(self) -> Optional[float]:
    #     """获取肺功能 FVC 值"""
    #     pass
