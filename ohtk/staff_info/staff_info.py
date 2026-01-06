import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel
from functional import seq
from datetime import datetime
from typing import Dict, Union
from loguru import logger

from ohtk.constants import AuditoryConstants
from ohtk.staff_info.staff_basic_info import StaffBasicInfo
from ohtk.staff_info.staff_health_info import StaffHealthInfo
from ohtk.staff_info.staff_occhaz_info import StaffOccHazInfo


class StaffInfo(BaseModel):
    staff_id: Union[str, int]
    staff_name: str = ""
    staff_sex: str = None
    staff_age: int = None
    staff_duration: int = None
    staff_basic_info: Dict[int, StaffBasicInfo] = {}
    staff_health_info: Dict[int, StaffHealthInfo] = {}
    staff_occhaz_info: Dict[int, StaffOccHazInfo] = {}
    conflict_attr: list = []
    record_years: list = []
    current_year: datetime = datetime.now().year

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)

    def _build(self, **data):
        # data中的信息应当是json形式，体检记录相关的字段下的类型为list
        record_mesg_df = seq(data.items()).filter(lambda x: x[0] in [
            "record_year", "name", "factory_name", "work_shop", "work_position", "sex", "age",
            "duration", "smoking", "year_of_smoking", "cigaretee_per_day",
            "occupational_clinic_class", "auditory_detection", "auditory_diagnose",
            "noise_hazard_info"]).dict()
        try:
            record_mesg_df = pd.DataFrame(
                record_mesg_df).set_index("record_year")
        except ValueError:
            logger.error("列表不等长，体检记录的内容存在缺失，请检查！")
            raise
        # 常量信息计算与校验
        staff_name = record_mesg_df["name"].value_counts()
        staff_sex = record_mesg_df["sex"].value_counts()
        staff_age = (
            record_mesg_df["age"] - record_mesg_df.index + self.current_year).value_counts()
        staff_duration = (
            record_mesg_df["duration"] - record_mesg_df.index + self.current_year).value_counts()
        for attr_name, attr in zip(["staff_name", "staff_sex", "staff_age", "staff_duration"],
                                   [staff_name, staff_sex, staff_age, staff_duration]):
            if len(attr) > 1:
                logger.warning(f"注意！根据记录得到的{attr_name}不唯一！将使用频数最高的结果。")
                self.conflict_attr.append(attr_name)

        self.staff_name = staff_name.idxmax()
        self.staff_sex = staff_sex.idxmax()
        self.staff_age = staff_age.idxmax()
        self.staff_duration = staff_duration.idxmax()

        # 体检记录信息逐条写入
        record_mesgs = record_mesg_df.to_dict(orient="index")
        staff_basic_info = {}
        staff_health_info = {}
        staff_occhaz_info = {}
        for record_year, record_mesg in record_mesgs.items():
            self.record_years.append(record_year)
            # 员工基础信息构建
            staff_basic_info[record_year] = StaffBasicInfo(staff_id=self.staff_id,
                                                           name=self.staff_name,
                                                           factory_name=record_mesg.get(
                                                               "factory_name"),
                                                           work_shop=record_mesg.get(
                                                               "work_shop"),
                                                           work_position=record_mesg.get(
                                                               "work_position"),
                                                           sex=self.staff_sex,
                                                           age=record_mesg.get("age"),
                                                           duration=record_mesg.get(
                                                               "duration"),
                                                           smoking=record_mesg.get(
                                                               "smoking"),
                                                           year_of_smoking=record_mesg.get(
                                                               "year_of_smoking"),
                                                           cigaretee_per_day=record_mesg.get(
                                                               "cigaretee_per_day"),
                                                           occupational_clinic_class=record_mesg.get("occupational_clinic_class"))
            # 员工健康诊断信息构建
            staff_health_info[record_year] = StaffHealthInfo(staff_id=self.staff_id,
                                                             sex=self.staff_sex,
                                                             age=record_mesg.get(
                                                                 "age"),
                                                             auditory_detection=record_mesg.get(
                                                                 "auditory_detection"),
                                                             auditory_diagnose=record_mesg.get("auditory_diagnose"))
            # 员工职业危害因素信息构建
            staff_occhaz_info[record_year] = StaffOccHazInfo(staff_id=self.staff_id,
                                                             noise_hazard_info=record_mesg.get("noise_hazard_info"))

        self.staff_basic_info = staff_basic_info
        self.staff_health_info = staff_health_info
        self.staff_occhaz_info = staff_occhaz_info

    def _get_staff_data_for_niots(self, **kwargs) -> tuple:
        """Helper method to get staff data for NIPTS prediction.
        
        Returns:
            tuple: (LAeq, age, sex, duration, remaining_kwargs)
        """
        # Get the most recent year's data
        recent_year = max(self.record_years) if self.record_years else self.current_year
        LAeq = kwargs.pop(
            "LAeq", None) or (self.staff_occhaz_info.get(recent_year).noise_hazard_info.LAeq if recent_year in self.staff_occhaz_info and self.staff_occhaz_info[recent_year].noise_hazard_info else None)
        age = kwargs.pop("age", None) or (self.staff_basic_info.get(recent_year).age if recent_year in self.staff_basic_info else self.staff_age)
        sex = kwargs.pop("sex", None) or (self.staff_basic_info.get(recent_year).sex if recent_year in self.staff_basic_info else self.staff_sex)
        duration = kwargs.pop(
            "duration", None) or (self.staff_basic_info.get(recent_year).duration if recent_year in self.staff_basic_info else self.staff_duration)
        
        return LAeq, age, sex, duration, kwargs

    def NIPTS_predict_iso1999_2013(self,
                                   Hs: bool = False,
                                   percentrage: int = 50,
                                   mean_key: list = [3000, 4000, 6000],
                                   standard: str = "Chinese",
                                   **kwargs) -> float:
        """根据ISO 1999:2013标准预测噪声性永久阈移(NIPTS)
        
        Args:
            Hs: 是否考虑基底听力损失
            percentrage: 百分位数，默认50
            mean_key: 频率列表，默认[3000, 4000, 6000]
            standard: 标准类型，"Chinese"或其他
            **kwargs: 可选参数，如LAeq, age, sex, duration, NH_limit
            
        Returns:
            float: 预测的NIPTS值
            
        Raises:
            ValueError: 当缺少必要数据时
        """
        # Get staff data using helper method
        LAeq, age, sex, duration, remaining_kwargs = self._get_staff_data_for_niots(**kwargs)
        NH_limit = remaining_kwargs.pop("NH_limit", True)

        # Check if required data is available
        if LAeq is None or age is None or sex is None or duration is None:
            raise ValueError("Required data (LAeq, age, sex, or duration) is missing for NIPTS prediction")

        sex = "Male" if sex.startswith("M") else "Female"
        age = AuditoryConstants.AGE_BOXING(age=age)
        percentrage = str(percentrage) + "pr"

        if standard == "Chinese":
            standard_PTA = AuditoryConstants.CHINESE_STANDARD_PTA_DICT.get(
                sex).get(age)
        else:
            standard_PTA = AuditoryConstants.ISO_1999_2013_STANDARD_PTA_DICT.get(
                sex).get(age)
        
        if standard_PTA is None:
            raise ValueError(f"Standard PTA data not found for sex: {sex}, age: {age}")
            
        standard_PTA = seq(standard_PTA.items()).filter(lambda x: int(x[0].split(
            "Hz")[0]) in mean_key).map(lambda x: (int(x[0].split("Hz")[0]), x[1])).dict()
        standard_PTA = seq(standard_PTA.items()).map(
            lambda x: (x[0], x[1].get(percentrage))).dict()

        NIPTS_preds = []
        for freq in mean_key:
            u = AuditoryConstants.ISO_1999_2013_NIPTS_PRED_DICT.get(
                str(freq) + "Hz").get("u")
            v = AuditoryConstants.ISO_1999_2013_NIPTS_PRED_DICT.get(
                str(freq) + "Hz").get("v")
            L0 = AuditoryConstants.ISO_1999_2013_NIPTS_PRED_DICT.get(
                str(freq) + "Hz").get("L0")
                
            if u is None or v is None or L0 is None:
                logger.warning(f"Missing parameter for frequency {freq}Hz, skipping...")
                continue
                
            if duration < 10:
                NIPTS_pred = np.log10(duration + 1) / np.log10(11) * (
                    u + v * np.log10(10 / 1)) * (LAeq - L0)**2
            else:
                NIPTS_pred = (u + v * np.log10(duration / 1)) * (LAeq - L0)**2

            if Hs:
                H = standard_PTA.get(freq)
                if H is not None:
                    if NH_limit:
                        if H + NIPTS_pred > 40:
                            NIPTS_pred = NIPTS_pred - NIPTS_pred * H / 120
                    else:
                        NIPTS_pred = NIPTS_pred - NIPTS_pred * H / 120
            NIPTS_preds.append(NIPTS_pred)
            
        if not NIPTS_preds:
            raise ValueError("No valid predictions could be calculated")
            
        NIPTS_pred_res = np.mean(NIPTS_preds)
        return NIPTS_pred_res

    def NIPTS_predict_iso1999_2023(self,
                                   percentrage: int = 50,
                                   mean_key: list = [3000, 4000, 6000],
                                   **kwargs) -> float:
        """根据ISO 1999:2023标准预测噪声性永久阈移(NIPTS)
        
        Args:
            percentrage: 百分位数，默认50
            mean_key: 频率列表，默认[3000, 4000, 6000]
            **kwargs: 可选参数，如LAeq, age, sex, duration, extrapolation, NH_limit
            
        Returns:
            float: 预测的NIPTS值
            
        Raises:
            ValueError: 当缺少必要数据时
        """
        # Get staff data using helper method
        LAeq, age, sex, duration, remaining_kwargs = self._get_staff_data_for_niots(**kwargs)
        extrapolation = remaining_kwargs.pop("extrapolation", None)
        NH_limit = remaining_kwargs.pop("NH_limit", True)

        # Check if required data is available
        if LAeq is None or age is None or sex is None or duration is None:
            raise ValueError("Required data (LAeq, age, sex, or duration) is missing for NIPTS prediction")

        # 标准化输入参数
        age = 21 if age <= 20 else age
        age = 70 if age > 70 else age
        duration = 40 if duration > 40 else duration
        duration = age - 20 if age - duration < 20 else duration
        sex_str = "Male" if sex == "M" else "Female"
        
        # 获取百分位数索引
        p_idx, p_next_idx, p_ratio = self._get_percentile_indices(percentrage)
        
        # 获取插值参数
        age_idx, age_next_idx, age_ratio = self._get_age_interpolation_params(age)
        laeq_idx, laeq_next_idx, laeq_ratio = self._get_laeq_interpolation_params(LAeq)
        duration_idx, duration_next_idx, duration_ratio = self._get_duration_interpolation_params(duration)

        # 计算预测值
        nipts_predictions = []
        for freq in mean_key:
            nipts_pred = self._calculate_nipts_for_frequency(
                freq, duration, laeq_idx, laeq_next_idx, laeq_ratio,
                duration_idx, duration_next_idx, duration_ratio,
                p_idx, p_next_idx, p_ratio, age_idx, age_next_idx, age_ratio,
                age, sex, LAeq, extrapolation, NH_limit, percentrage, mean_key
            )
            if not np.isnan(nipts_pred):
                nipts_predictions.append(nipts_pred)

        if not nipts_predictions:
            raise ValueError("No valid predictions could be calculated")
            
        return np.mean(nipts_predictions)


    def _get_percentile_indices(self, percentrage: int) -> tuple:
        """获取百分位数的插值参数"""
        if 90 <= percentrage <= 95:
            p_idx = 0
        elif 75 <= percentrage < 90:
            p_idx = 1
        elif 50 <= percentrage < 75:
            p_idx = 2
        elif 25 <= percentrage < 50:
            p_idx = 3
        elif 10 <= percentrage < 25:
            p_idx = 4
        elif 5 <= percentrage < 10:
            p_idx = 5
        else:
            raise ValueError(f"Percentrage {percentrage} is out of valid range (5-95)")
        
        p_next_idx = p_idx + 1
        ps_values = [90, 95, 75, 50, 25, 10, 5]
        p_ratio = (percentrage - ps_values[p_idx]) / (ps_values[p_next_idx] - ps_values[p_idx])
        return p_idx, p_next_idx, p_ratio

    def _get_age_interpolation_params(self, age: int) -> tuple:
        """获取年龄的插值参数"""
        As = [20, 30, 40, 50, 60]
        A0 = (age - 10) / 10
        A1 = int(A0) if age < 70 else 5
        A1 = min(A1, len(As)-2)  # 确保不超过As的最大索引
        A2 = A1 + 1
        AR = (A0 - A1) / (A2 - A1)
        return A1, A2, AR

    def _get_laeq_interpolation_params(self, LAeq: float) -> tuple:
        """获取LAeq的插值参数"""
        ls = [70, 75, 80, 85, 90, 95, 100]
        L0 = (LAeq - 65) / 5
        L1 = int(L0) if LAeq < 100 else 6
        L1 = min(L1, len(ls)-2)  # 确保不超过ls的最大索引
        L2 = L1 + 1
        LR = (L0 - L1) / (L2 - L1)
        return L1, L2, LR

    def _get_duration_interpolation_params(self, duration: float) -> tuple:
        """获取工龄的插值参数"""
        D0 = duration / 10
        D1 = int(D0) if duration != 40 else 3
        D1 = D1 if duration >= 10 else 1  # 对于小于10年的情况
        D2 = D1 + 1
        DR = (D0 - D1) / (D2 - D1)
        return D1, D2, DR

    def _dict_query_nipts(self, laeq_idx: int, duration_idx: int, p_idx: int, freq: int) -> float:
        """查询NIPTS预测字典表"""
        ls = [70, 75, 80, 85, 90, 95, 100]
        ps = [90, 95, 75, 50, 25, 10, 5]
        
        laeq_str = f"{ls[laeq_idx]}dB"
        duration_str = f"{(duration_idx + 1) * 10}years"
        percentage_str = f"{ps[p_idx]}pr"
        freq_str = f"{freq}Hz"

        try:
            dict_1 = AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_T1.get(duration_str)
            return dict_1.get(laeq_str).get(freq_str).get(percentage_str)
        except (AttributeError, KeyError, TypeError):
            return np.nan

    def _dict_query_age_hearing(self, age_idx: int, sex: str, p_idx: int, freq: int) -> float:
        """查询年龄相关的听力损失字典表"""
        As = [20, 30, 40, 50, 60]
        ps = [90, 95, 75, 50, 25, 10, 5]
        
        age_str = str(As[age_idx])
        percentage_str = f"{ps[p_idx]}pr"
        freq_str = f"{freq}Hz"

        try:
            dict_1 = AuditoryConstants.ISO_1999_2023_NIPTS_PRED_DICT_B6.get(sex)
            return dict_1.get(age_str).get(freq_str).get(percentage_str, 0)
        except (AttributeError, KeyError, TypeError):
            return np.nan

    def _calculate_nipts_for_frequency(self, freq: int, duration: int, 
                                      laeq_idx: int, laeq_next_idx: int, laeq_ratio: float,
                                      duration_idx: int, duration_next_idx: int, duration_ratio: float,
                                      p_idx: int, p_next_idx: int, p_ratio: float,
                                      age_idx: int, age_next_idx: int, age_ratio: float,
                                      age: int, sex: str, LAeq: float, 
                                      extrapolation: str, NH_limit: bool, 
                                      percentrage: int, mean_key: list) -> float:
        """计算指定频率下的NIPTS值"""
        try:
            if duration < 10:
                LG = int(np.log10(duration + 1) / np.log10(11) * 10 + 0.5) / 10
                NQ1 = int((self._dict_query_nipts(laeq_idx, 0, p_idx, freq) + laeq_ratio *
                           (self._dict_query_nipts(laeq_next_idx, 0, p_idx, freq) - 
                            self._dict_query_nipts(laeq_idx, 0, p_idx, freq))) * 10 + 0.5) / 10
                NQ2 = int((self._dict_query_nipts(laeq_idx, 0, p_next_idx, freq) + laeq_ratio *
                           (self._dict_query_nipts(laeq_next_idx, 0, p_next_idx, freq) -
                            self._dict_query_nipts(laeq_idx, 0, p_next_idx, freq))) * 10 + 0.5) / 10
                nipts_ldq = LG * int(((NQ1 + p_ratio * (NQ2 - NQ1))) * 10 + 0.5) / 10
            else:
                n1 = int((self._dict_query_nipts(laeq_idx, duration_idx, p_idx, freq) + laeq_ratio *
                          (self._dict_query_nipts(laeq_next_idx, duration_idx, p_idx, freq) -
                           self._dict_query_nipts(laeq_idx, duration_idx, p_idx, freq))) * 10 + 0.5) / 10
                n2 = int((self._dict_query_nipts(laeq_idx, duration_next_idx, p_idx, freq) + laeq_ratio *
                          (self._dict_query_nipts(laeq_next_idx, duration_next_idx, p_idx, freq) -
                           self._dict_query_nipts(laeq_idx, duration_next_idx, p_idx, freq))) * 10 + 0.5) / 10
                NQ1 = int(((n1 + duration_ratio * (n2 - n1))) * 10 + 0.5) / 10
                n1 = int((self._dict_query_nipts(laeq_idx, duration_idx, p_next_idx, freq) + laeq_ratio *
                          (self._dict_query_nipts(laeq_next_idx, duration_idx, p_next_idx, freq) -
                           self._dict_query_nipts(laeq_idx, duration_idx, p_next_idx, freq))) * 10 + 0.5) / 10
                n2 = int((self._dict_query_nipts(laeq_idx, duration_next_idx, p_next_idx, freq) + laeq_ratio *
                          (self._dict_query_nipts(laeq_next_idx, duration_next_idx, p_next_idx, freq) -
                           self._dict_query_nipts(laeq_idx, duration_next_idx, p_next_idx, freq))) * 10 + 0.5) / 10
                NQ2 = int(((n1 + duration_ratio * (n2 - n1))) * 10 + 0.5) / 10
                nipts_ldq = int((NQ1 + p_ratio * (NQ2 - NQ1)) * 10 + 0.5) / 10
        except (ValueError, TypeError, AttributeError):
            nipts_ldq = np.nan

        try:
            H1 = int((self._dict_query_age_hearing(age_idx, sex, p_idx, freq) + age_ratio *
                      (self._dict_query_age_hearing(age_next_idx, sex, p_idx, freq) -
                       self._dict_query_age_hearing(age_idx, sex, p_idx, freq))) * 10 + 0.5) / 10
            H2 = int((self._dict_query_age_hearing(age_idx, sex, p_next_idx, freq) + age_ratio *
                      (self._dict_query_age_hearing(age_next_idx, sex, p_next_idx, freq) -
                       self._dict_query_age_hearing(age_idx, sex, p_next_idx, freq))) * 10 + 0.5) / 10
            H = int(((H1 + p_ratio * (H2 - H1))) * 10 + 0.5) / 10
        except (ValueError, TypeError, AttributeError):
            H = np.nan

        # 检查边界条件
        if age < 20 or age > 70:
            nipts_ldq = np.nan
        if duration < 1 or duration > 40 or age - duration < 20:
            nipts_ldq = np.nan
        if LAeq < 70:
            nipts_ldq = np.nan
        if LAeq > 100:
            nipts_ldq = self._handle_extrapolation(LAeq, age, duration, sex, freq, percentrage, mean_key, extrapolation, H, NH_limit)
            if np.isnan(nipts_ldq):
                return np.nan

        # 应用NH_limit限制
        if NH_limit:
            if H + nipts_ldq > 40:
                nipts_ldq = nipts_ldq - H * nipts_ldq / 120
        else:
            nipts_ldq = nipts_ldq - H * nipts_ldq / 120

        return nipts_ldq

    def _handle_extrapolation(self, LAeq: float, age: int, duration: int, sex: str, freq: int, percentrage: int, mean_key: list, 
                             extrapolation: str, H: float, NH_limit: bool) -> float:
        """处理外推情况"""
        if extrapolation == "ML":
            try:
                model = pickle.load(
                    open(
                        f"./model/regression_model_for_NIPTS_pred_2023.pkl",
                        "rb"))
                feature = [[1 if sex == "M" else 0, age, duration, LAeq]]
                return model.predict(
                    pd.DataFrame(
                        feature,
                        columns=["sex_encoder", "age", "duration",
                                 "LAeq"]))[0]
            except Exception as e:
                logger.error(f"ML extrapolation failed: {e}")
                return np.nan
        elif extrapolation == "Linear":
            try:
                # 使用标准表上限 (100 dB) 和次高点 (95 dB) 进行线性外推
                nipts_pred_95 = self.NIPTS_predict_iso1999_2023(
                    LAeq=95, percentrage=percentrage, mean_key=mean_key,
                    extrapolation=None, NH_limit=NH_limit)
                nipts_pred_100 = self.NIPTS_predict_iso1999_2023(
                    LAeq=100, percentrage=percentrage, mean_key=mean_key,
                    extrapolation=None, NH_limit=NH_limit)
                # 计算斜率
                slope = (nipts_pred_100 - nipts_pred_95) / 5
                # 使用点斜式: y - y1 = slope * (x - x1), 其中 (x1,y1) 是 (100, nipts_pred_100)
                return nipts_pred_100 + slope * (LAeq - 100)
            except RecursionError:
                logger.error("Recursion error during extrapolation, returning NaN")
                return np.nan
        else:
            return np.nan
