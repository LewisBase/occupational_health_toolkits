# -*- coding: utf-8 -*-
"""
体检记录基础信息模块

StaffBasicInfo 存储每次体检记录的特定信息。
注意：sex/age/duration 由 StaffInfo 统一存储，此处不重复存储。
"""

from pydantic import BaseModel, model_validator
from typing import Union, Optional
from datetime import datetime
import pandas as pd


class StaffBasicInfo(BaseModel):
    """
    体检记录基础信息
    
    存储每次体检记录的特定信息，如时间、地点、工作单位等。
    
    注意：
    - sex/age/duration 从 StaffInfo 获取，此处不存储
    - creation_date 和 creation_province 为必需字段
    """
    staff_id: Union[int, str]                                # 员工ID
    
    # === 必需字段（记录特定） ===
    creation_date: datetime                                  # 检查日期（必需）
    creation_province: str                                   # 省份（必需）
    
    # === 可选字段 ===
    # 个人信息
    name: Optional[str] = None                               # 姓名
    
    # 工作信息
    factory_name: Optional[str] = None                       # 工厂名称
    work_shop: Optional[Union[str, float]] = None            # 车间
    work_position: Optional[Union[str, float]] = None        # 工种/岗位
    
    # 地点信息
    creation_city: Optional[str] = None                      # 城市
    industry_category: Optional[str] = None                  # 行业类别
    employment_unit_code: Optional[str] = None               # 单位统一信用代码
    examination_type_code: Optional[int] = None              # 体检类型编码 (1-5)
    
    # 吸烟信息
    smoking: Optional[Union[str, int]] = None                # 是否抽烟
    year_of_smoking: Optional[Union[float, str, int]] = None # 烟龄
    cigarette_per_day: Optional[Union[float, str, int]] = None  # 平均每天抽烟数量
    occupational_clinic_class: Optional[str] = None          # 职业健康体检类型

    @model_validator(mode='before')
    @classmethod
    def parse_creation_date(cls, data: dict) -> dict:
        """在验证前解析日期字符串"""
        if isinstance(data, dict) and 'creation_date' in data:
            val = data['creation_date']
            if isinstance(val, str):
                try:
                    data['creation_date'] = pd.to_datetime(val)
                except Exception:
                    pass
        return data

    @model_validator(mode='after')
    def validate_basic_info(self) -> 'StaffBasicInfo':
        """验证和处理基础信息"""
        # 处理吸烟信息
        if self.smoking is None or self.smoking == "N" or self.smoking == 0:
            self.smoking = "N"
            if self.year_of_smoking is None:
                self.year_of_smoking = 0
            if self.cigarette_per_day is None:
                self.cigarette_per_day = 0
        
        return self
