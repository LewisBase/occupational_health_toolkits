import numpy as np
from functional import seq
from pydantic import BaseModel
from typing import Union

from ohtk.constants.auditory_constants import AuditoryConstants
from ohtk.detection_info.auditory_detection import PTAResult


class AuditoryDiagnose(BaseModel):
    # NIPTS: float = None
    # is_NIHL: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **kwargs):
        pass

    @staticmethod
    def NIPTS(detection_result: PTAResult, # type: ignore
              sex: str, age: int,
              percentrage: int = 50,
              mean_key: Union[list, dict] = [3000, 4000, 6000],
              NIPTS_diagnose_strategy: str = "better",
              standard: str = "Chinese",
              **kwargs):
        if NIPTS_diagnose_strategy == "better":
            diagnose_ear_data = detection_result.better_ear_data
        elif NIPTS_diagnose_strategy == "left":
            diagnose_ear_data = detection_result.left_ear_data
        elif NIPTS_diagnose_strategy == "right":
            diagnose_ear_data = detection_result.right_ear_data
        elif NIPTS_diagnose_strategy == "poorer":
            diagnose_ear_data = detection_result.poorer_ear_data
        elif NIPTS_diagnose_strategy == "mean":
            diagnose_ear_data = detection_result.mean_ear_data
        else:
            raise ValueError("NIPTS_diagnose_strategy must be one of 'better', 'left', 'right', 'poorer', 'mean'")

        sex = "Male" if sex in ("Male", "男", "M", "m", "male") else "Female"
        age = AuditoryConstants.AGE_BOXING(age=age, standard=standard, sex=sex)
        percentrage = str(percentrage) + "pr"
        if standard == "Chinese":
            standard_PTA = AuditoryConstants.CHINESE_STANDARD_PTA_DICT.get(sex).get(age)
        elif standard == "ISO":
            standard_PTA = AuditoryConstants.ISO_1999_2013_STANDARD_DICT.get(sex).get(age)
        elif standard == "NIOSH_paper":
            standard_PTA = AuditoryConstants.NIOSH_paper_STANDARD_DICT.get(sex).get(age)
            
        if isinstance(mean_key, list):
            standard_PTA = seq(standard_PTA.items()).filter(lambda x: int(x[0].split(
                "Hz")[0]) in mean_key).map(lambda x: (int(x[0].split("Hz")[0]), x[1])).dict()
        if isinstance(mean_key, dict):
            standard_PTA = seq(standard_PTA.items()).filter(lambda x: int(x[0].split(
                "Hz")[0]) in mean_key.keys()).map(lambda x: (int(x[0].split("Hz")[0]), x[1])).dict()
        standard_PTA = seq(standard_PTA.items()).map(
            lambda x: (x[0], x[1].get(percentrage))).dict()

        try:
            if isinstance(mean_key, list):
                NIPTS = np.mean([diagnose_ear_data.get(key) -
                                 standard_PTA.get(key) for key in mean_key])
            if isinstance(mean_key, dict):
                NIPTS = np.mean([diagnose_ear_data.get(key) -
                                 standard_PTA.get(key) for key in mean_key.keys()])
        except TypeError:
            raise TypeError("Better ear data is incompleted!!!")
        return NIPTS
