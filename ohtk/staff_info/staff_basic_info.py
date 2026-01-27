from pydantic import BaseModel
from typing import Union, Optional


class StaffBasicInfo(BaseModel):
    staff_id: Union[int, str]                      # 工厂名称+自增id
    factory_name: str                              # 工厂名称
    work_shop: Union[str, float]                   # 车间
    work_position: Union[str, float]               # 工种/岗位
    name: str = ""                                 # 姓名
    sex: str                                       # 性别
    age: Union[int,float]                          # 年龄
    duration: float                                # 工龄，单位年
    smoking: Optional[Union[str, int]] = None                # 是否抽烟
    year_of_smoking: Optional[Union[float, str]] = None      # 烟龄
    cigarette_per_day: Optional[Union[float, str]] = None    # 平均每天抽烟数量
    occupational_clinic_class: Optional[str] = None          # 职业健康体检类型

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **data):
        self.sex = "M" if self.sex in ("Male", "男", "M", "m", "male") else "F"
        if self.smoking is None or self.smoking == "N":
            self.smoking = "N"
            self.year_of_smoking = 0
            self.cigarette_per_day = 0
