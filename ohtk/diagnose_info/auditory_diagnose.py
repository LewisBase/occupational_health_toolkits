import numpy as np
import pickle
import logging
from functional import seq
from pydantic import BaseModel
from typing import Union, Dict, List, Optional

from ohtk.constants.auditory_constants import AuditoryConstants
from ohtk.detection_info.auditory_detection import PTAResult
from ohtk.utils.pta_correction import calculate_nihl, calculate_all_nihl

logger = logging.getLogger(__name__)


class AuditoryDiagnose(BaseModel):
    """
    听力诊断类 - 提供 NIHL 计算和 NIPTS 分析的静态方法
    
    【重要说明】此类中的方法将逐步废弃，建议使用新架构：
    - NIHL 计算 → 使用 ohtk.diagnose_info.nihl_predictor
    - NIPTS 预测 → 使用 ohtk.diagnose_info.nipts_predictor
    
    主要功能:
    - calculate_observed_NIPTS(): 计算观测到的实际NIPTS（从PTA减去年龄标准值）
    - calculate_NIHL(): 计算噪声性听力损失指标
    - calculate_all_NIHL(): 计算所有频率组合的NIHL
    - predict_NIPTS_iso1999_2013(): 基于 ISO 1999:2013 预测 NIPTS
    - predict_NIPTS_iso1999_2023(): 基于 ISO 1999:2023 预测 NIPTS
    
    【已废弃的方法名】:
    - NIPTS() → 请使用 calculate_observed_NIPTS()
    - NIHL() → 请使用 calculate_NIHL()
    - NIHL_all() → 请使用 calculate_all_NIHL()
    """

    @staticmethod
    def calculate_observed_NIPTS(detection_result: PTAResult, # type: ignore
                                  sex: str, age: int,
                                  percentrage: int = 50,
                                  mean_key: Union[list, dict] = [3000, 4000, 6000],
                                  NIPTS_diagnose_strategy: str = "better",
                                  standard: str = "Chinese",
                                  **kwargs):
        """
        计算观测到的 NIPTS（噪声性永久阈移）
        
        NIPTS_diagnose_strategy 参数说明：
        - "better": 使用 optimum_ear_data（每个频率独立取更好耳，标准 NIPTS 计算方式）
        - "left": 使用左耳数据
        - "right": 使用右耳数据  
        - "poorer": 使用 poorer_ear_data（整体较差耳）
        - "mean": 使用 mean_ear_data（两耳平均）
        """
        if NIPTS_diagnose_strategy == "better":
            # 使用 optimum_ear_data：每个频率独立取更好耳的值
            # 这是 NIPTS 观测值的标准计算方式
            diagnose_ear_data = detection_result.optimum_ear_data
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
    

    @staticmethod
    def calculate_NIHL(
        ear_data: Union[Dict[str, float], "PTAResult"],
        freq_key: str = "346",
        age: Optional[int] = None,
        sex: Optional[str] = None,
        apply_correction: bool = False,
        use_better_ear: bool = True
    ) -> float:
        """
        计算噪声性听力损失(NIHL)指标
        
        支持两种输入格式：
        1. 字典格式：调用 pta_correction.calculate_nihl() 计算双耳平均
        2. PTAResult 对象：使用 better_ear_data 计算更好耳平均
        
        Args:
            ear_data: 双耳听阈数据，支持两种格式：
                - Dict: {'left_ear_3000': 25.0, 'right_ear_4000': 30.0, ...}
                - PTAResult: PTAResult 对象
            freq_key: 频率配置键
                - "1234": 1000, 2000, 3000, 4000 Hz (言语频率)
                - "346": 3000, 4000, 6000 Hz (高频)
            age: 年龄（用于年龄校正时必需）
            sex: 性别 'M' 或 'F'（用于年龄校正时必需）
            apply_correction: 是否应用年龄校正
            use_better_ear: 当输入为 PTAResult 时，是否使用更好耳数据（默认 True）
            
        Returns:
            NIHL 值 (dB)，若数据不足则返回 np.nan
        """
        from ohtk.detection_info.auditory_detection.PTA_result import PTAResult
        from ohtk.utils.pta_correction import NIHL_FREQ_CONFIG
        
        # 判断输入类型
        if isinstance(ear_data, PTAResult):
            # 使用 PTAResult 对象
            frequencies = NIHL_FREQ_CONFIG.get(freq_key, [3000, 4000, 6000])
            if use_better_ear:
                ear_dict = ear_data.better_ear_data or {}
            else:
                # 合并左右耳数据
                ear_dict = {}
                if ear_data.left_ear_data:
                    ear_dict.update({f"left_ear_{k}": v for k, v in ear_data.left_ear_data.items()})
                if ear_data.right_ear_data:
                    ear_dict.update({f"right_ear_{k}": v for k, v in ear_data.right_ear_data.items()})
            
            # 提取指定频率的值
            values = []
            for freq in frequencies:
                # PTAResult 的 better_ear_data 使用频率作为 key
                if freq in ear_dict:
                    value = ear_dict[freq]
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        if apply_correction and age is not None and sex is not None:
                            from ohtk.utils.pta_correction import correct_pta_value
                            value = correct_pta_value(value, age, sex, freq)
                        values.append(value)
            
            return float(np.mean(values)) if values else np.nan
        else:
            # 使用字典格式
            return calculate_nihl(
                ear_data=ear_data,
                freq_key=freq_key,
                age=age,
                sex=sex,
                apply_correction=apply_correction
            )


    @staticmethod
    def calculate_all_NIHL(
        ear_data: Dict[str, float],
        age: Optional[int] = None,
        sex: Optional[str] = None,
        apply_correction: bool = False
    ) -> Dict[str, float]:
        """
        计算所有 NIHL 指标（1234 和 346）
        
        Args:
            ear_data: 双耳听阈数据
            age: 年龄
            sex: 性别
            apply_correction: 是否应用年龄校正
            
        Returns:
            {"1234": float, "346": float}
        """
        return calculate_all_nihl(
            ear_data=ear_data,
            age=age,
            sex=sex,
            apply_correction=apply_correction
        )
    

    @staticmethod
    def predict_NIPTS_iso1999_2013(
        LAeq: float,
        age: int,
        sex: str,
        duration: float,
        Hs: bool = False,
        percentrage: int = 50,
        mean_key: List[int] = None,
        standard: str = "Chinese",
        NH_limit: bool = True
    ) -> float:
        """
        根据 ISO 1999:2013 标准预测噪声性永久阈移(NIPTS)
        
        【注意】此方法已废弃，建议使用新架构：
        from ohtk.diagnose_info.nipts_predictor import get_predictor
        predictor = get_predictor('iso1999_2013')
        result = predictor.predict(LAeq=..., age=..., sex=..., duration=...)
        
        Args:
            LAeq: 等效连续A计权声压级 (dB)
            age: 年龄
            sex: 性别 ('M'/'F' 或其他格式)
            duration: 接噪工龄 (年)
            Hs: 是否考虑基底听力损失
            percentrage: 百分位数，默认 50
            mean_key: 频率列表，默认 [3000, 4000, 6000]
            standard: 标准类型，"Chinese" 或其他
            NH_limit: 是否限制非噪声相关听力损失
            
        Returns:
            预测的 NIPTS 值 (dB)
            
        Raises:
            ValueError: 当缺少必要数据或无法计算时
        """
        from ohtk.diagnose_info.nipts_predictor import get_predictor
        
        predictor = get_predictor('iso1999_2013')
        result = predictor.predict(
            LAeq=LAeq, age=age, sex=sex, duration=duration,
            percentrage=percentrage, mean_key=mean_key,
            Hs=Hs, standard=standard, NH_limit=NH_limit
        )
        return result.value

    @staticmethod
    def predict_NIPTS_iso1999_2023(
        LAeq: float,
        age: int,
        sex: str,
        duration: float,
        percentrage: int = 50,
        mean_key: List[int] = None,
        extrapolation: Optional[str] = None,
        NH_limit: bool = True
    ) -> float:
        """
        根据 ISO 1999:2023 标准预测噪声性永久阈移(NIPTS)
        
        【注意】此方法已废弃，建议使用新架构：
        from ohtk.diagnose_info.nipts_predictor import get_predictor
        predictor = get_predictor('iso1999_2023')
        result = predictor.predict(LAeq=..., age=..., sex=..., duration=...)
        
        Args:
            LAeq: 等效连续A计权声压级 (dB)
            age: 年龄
            sex: 性别 ('M'/'F' 或其他格式)
            duration: 接噪工龄 (年)
            percentrage: 百分位数，默认 50
            mean_key: 频率列表，默认 [3000, 4000, 6000]
            extrapolation: 外推方法 ("ML" 或 "Linear")，当 LAeq > 100 时使用
            NH_limit: 是否限制非噪声相关听力损失
            
        Returns:
            预测的 NIPTS 值 (dB)
            
        Raises:
            ValueError: 当缺少必要数据或无法计算时
        """
        from ohtk.diagnose_info.nipts_predictor import get_predictor
        
        predictor = get_predictor('iso1999_2023')
        result = predictor.predict(
            LAeq=LAeq, age=age, sex=sex, duration=duration,
            percentrage=percentrage, mean_key=mean_key,
            extrapolation=extrapolation, NH_limit=NH_limit
        )
        return result.value

    # ============ 私有静态辅助方法 ============

    @staticmethod
    def _get_percentile_indices(percentrage: int) -> tuple:
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

    @staticmethod
    def _get_age_interpolation_params(age: int) -> tuple:
        """获取年龄的插值参数"""
        As = [20, 30, 40, 50, 60]
        A0 = (age - 10) / 10
        A1 = int(A0) if age < 70 else 5
        A1 = min(A1, len(As) - 2)
        A2 = A1 + 1
        AR = (A0 - A1) / (A2 - A1)
        return A1, A2, AR

    @staticmethod
    def _get_laeq_interpolation_params(LAeq: float) -> tuple:
        """获取 LAeq 的插值参数"""
        ls = [70, 75, 80, 85, 90, 95, 100]
        L0 = (LAeq - 65) / 5
        L1 = int(L0) if LAeq < 100 else 6
        L1 = min(L1, len(ls) - 2)
        L2 = L1 + 1
        LR = (L0 - L1) / (L2 - L1)
        return L1, L2, LR

    @staticmethod
    def _get_duration_interpolation_params(duration: float) -> tuple:
        """获取工龄的插值参数"""
        D0 = duration / 10
        D1 = int(D0) if duration != 40 else 3
        D1 = D1 if duration >= 10 else 1
        D2 = D1 + 1
        DR = (D0 - D1) / (D2 - D1)
        return D1, D2, DR

    @staticmethod
    def _dict_query_nipts(laeq_idx: int, duration_idx: int, p_idx: int, freq: int) -> float:
        """查询 NIPTS 预测字典表"""
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

    @staticmethod
    def _dict_query_age_hearing(age_idx: int, sex: str, p_idx: int, freq: int) -> float:
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

    @staticmethod
    def _calculate_nipts_for_frequency(
        freq: int, duration: float,
        laeq_idx: int, laeq_next_idx: int, laeq_ratio: float,
        duration_idx: int, duration_next_idx: int, duration_ratio: float,
        p_idx: int, p_next_idx: int, p_ratio: float,
        age_idx: int, age_next_idx: int, age_ratio: float,
        age: int, sex: str, LAeq: float,
        extrapolation: Optional[str], NH_limit: bool,
        percentrage: int, mean_key: List[int]
    ) -> float:
        """计算指定频率下的 NIPTS 值"""
        try:
            if duration < 10:
                LG = int(np.log10(duration + 1) / np.log10(11) * 10 + 0.5) / 10
                NQ1 = int((AuditoryDiagnose._dict_query_nipts(laeq_idx, 0, p_idx, freq) + laeq_ratio *
                           (AuditoryDiagnose._dict_query_nipts(laeq_next_idx, 0, p_idx, freq) - 
                            AuditoryDiagnose._dict_query_nipts(laeq_idx, 0, p_idx, freq))) * 10 + 0.5) / 10
                NQ2 = int((AuditoryDiagnose._dict_query_nipts(laeq_idx, 0, p_next_idx, freq) + laeq_ratio *
                           (AuditoryDiagnose._dict_query_nipts(laeq_next_idx, 0, p_next_idx, freq) -
                            AuditoryDiagnose._dict_query_nipts(laeq_idx, 0, p_next_idx, freq))) * 10 + 0.5) / 10
                nipts_ldq = LG * int(((NQ1 + p_ratio * (NQ2 - NQ1))) * 10 + 0.5) / 10
            else:
                n1 = int((AuditoryDiagnose._dict_query_nipts(laeq_idx, duration_idx, p_idx, freq) + laeq_ratio *
                          (AuditoryDiagnose._dict_query_nipts(laeq_next_idx, duration_idx, p_idx, freq) -
                           AuditoryDiagnose._dict_query_nipts(laeq_idx, duration_idx, p_idx, freq))) * 10 + 0.5) / 10
                n2 = int((AuditoryDiagnose._dict_query_nipts(laeq_idx, duration_next_idx, p_idx, freq) + laeq_ratio *
                          (AuditoryDiagnose._dict_query_nipts(laeq_next_idx, duration_next_idx, p_idx, freq) -
                           AuditoryDiagnose._dict_query_nipts(laeq_idx, duration_next_idx, p_idx, freq))) * 10 + 0.5) / 10
                NQ1 = int(((n1 + duration_ratio * (n2 - n1))) * 10 + 0.5) / 10
                n1 = int((AuditoryDiagnose._dict_query_nipts(laeq_idx, duration_idx, p_next_idx, freq) + laeq_ratio *
                          (AuditoryDiagnose._dict_query_nipts(laeq_next_idx, duration_idx, p_next_idx, freq) -
                           AuditoryDiagnose._dict_query_nipts(laeq_idx, duration_idx, p_next_idx, freq))) * 10 + 0.5) / 10
                n2 = int((AuditoryDiagnose._dict_query_nipts(laeq_idx, duration_next_idx, p_next_idx, freq) + laeq_ratio *
                          (AuditoryDiagnose._dict_query_nipts(laeq_next_idx, duration_next_idx, p_next_idx, freq) -
                           AuditoryDiagnose._dict_query_nipts(laeq_idx, duration_next_idx, p_next_idx, freq))) * 10 + 0.5) / 10
                NQ2 = int(((n1 + duration_ratio * (n2 - n1))) * 10 + 0.5) / 10
                nipts_ldq = int((NQ1 + p_ratio * (NQ2 - NQ1)) * 10 + 0.5) / 10
        except (ValueError, TypeError, AttributeError):
            nipts_ldq = np.nan

        try:
            H1 = int((AuditoryDiagnose._dict_query_age_hearing(age_idx, sex, p_idx, freq) + age_ratio *
                      (AuditoryDiagnose._dict_query_age_hearing(age_next_idx, sex, p_idx, freq) -
                       AuditoryDiagnose._dict_query_age_hearing(age_idx, sex, p_idx, freq))) * 10 + 0.5) / 10
            H2 = int((AuditoryDiagnose._dict_query_age_hearing(age_idx, sex, p_next_idx, freq) + age_ratio *
                      (AuditoryDiagnose._dict_query_age_hearing(age_next_idx, sex, p_next_idx, freq) -
                       AuditoryDiagnose._dict_query_age_hearing(age_idx, sex, p_next_idx, freq))) * 10 + 0.5) / 10
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
            nipts_ldq = AuditoryDiagnose._handle_extrapolation(
                LAeq, age, duration, sex, freq, percentrage, mean_key, extrapolation, H, NH_limit
            )
            if np.isnan(nipts_ldq):
                return np.nan

        # 应用 NH_limit 限制
        if not np.isnan(nipts_ldq) and not np.isnan(H):
            if NH_limit:
                if H + nipts_ldq > 40:
                    nipts_ldq = nipts_ldq - H * nipts_ldq / 120
            else:
                nipts_ldq = nipts_ldq - H * nipts_ldq / 120

        return nipts_ldq

    @staticmethod
    def _handle_extrapolation(
        LAeq: float, age: int, duration: float, sex: str, freq: int,
        percentrage: int, mean_key: List[int],
        extrapolation: Optional[str], H: float, NH_limit: bool
    ) -> float:
        """处理外推情况"""
        if extrapolation == "ML":
            try:
                import pandas as pd
                model = pickle.load(
                    open(f"./model/regression_model_for_NIPTS_pred_2023.pkl", "rb")
                )
                sex_encoded = 1 if sex == "Male" or str(sex).startswith("M") else 0
                feature = [[sex_encoded, age, duration, LAeq]]
                return model.predict(
                    pd.DataFrame(feature, columns=["sex_encoder", "age", "duration", "LAeq"])
                )[0]
            except Exception as e:
                logger.error(f"ML extrapolation failed: {e}")
                return np.nan
        elif extrapolation == "Linear":
            try:
                # 使用标准表上限 (100 dB) 和次高点 (95 dB) 进行线性外推
                nipts_pred_95 = AuditoryDiagnose.predict_NIPTS_iso1999_2023(
                    LAeq=95, age=age, sex=sex, duration=duration,
                    percentrage=percentrage, mean_key=mean_key,
                    extrapolation=None, NH_limit=NH_limit
                )
                nipts_pred_100 = AuditoryDiagnose.predict_NIPTS_iso1999_2023(
                    LAeq=100, age=age, sex=sex, duration=duration,
                    percentrage=percentrage, mean_key=mean_key,
                    extrapolation=None, NH_limit=NH_limit
                )
                slope = (nipts_pred_100 - nipts_pred_95) / 5
                return nipts_pred_100 + slope * (LAeq - 100)
            except RecursionError:
                logger.error("Recursion error during extrapolation, returning NaN")
                return np.nan
        else:
            return np.nan