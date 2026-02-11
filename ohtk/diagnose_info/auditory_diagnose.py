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
    
    @staticmethod
    def NIPTS(*args, **kwargs):
        """
        【已废弃】请使用 calculate_observed_NIPTS() 代替
        
        此方法保留用于向后兼容，将在未来版本中移除
        """
        logger.warning(
            "AuditoryDiagnose.NIPTS() 已废弃，请使用 calculate_observed_NIPTS() 代替"
        )
        return AuditoryDiagnose.calculate_observed_NIPTS(*args, **kwargs)

    @staticmethod
    def calculate_NIHL(
        ear_data: Dict[str, float],
        freq_key: str = "346",
        age: Optional[int] = None,
        sex: Optional[str] = None,
        apply_correction: bool = False
    ) -> float:
        """
        计算噪声性听力损失(NIHL)指标
        
        包装 pta_correction.calculate_nihl() 函数
        
        Args:
            ear_data: 双耳听阈数据，格式如 {'left_ear_3000': 25.0, 'right_ear_4000': 30.0, ...}
            freq_key: 频率配置键
                - "1234": 1000, 2000, 3000, 4000 Hz (言语频率)
                - "346": 3000, 4000, 6000 Hz (高频)
            age: 年龄（用于年龄校正时必需）
            sex: 性别 'M' 或 'F'（用于年龄校正时必需）
            apply_correction: 是否应用年龄校正
            
        Returns:
            NIHL 值 (dB)，若数据不足则返回 np.nan
        """
        return calculate_nihl(
            ear_data=ear_data,
            freq_key=freq_key,
            age=age,
            sex=sex,
            apply_correction=apply_correction
        )

    @staticmethod
    def NIHL(*args, **kwargs) -> float:
        """
        【已废弃】请使用 calculate_NIHL() 代替
        
        此方法保留用于向后兼容，将在未来版本中移除
        """
        logger.warning(
            "AuditoryDiagnose.NIHL() 已废弃，请使用 calculate_NIHL() 代替"
        )
        return AuditoryDiagnose.calculate_NIHL(*args, **kwargs)

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
    def NIHL_all(*args, **kwargs) -> Dict[str, float]:
        """
        【已废弃】请使用 calculate_all_NIHL() 代替
        
        此方法保留用于向后兼容，将在未来版本中移除
        """
        logger.warning(
            "AuditoryDiagnose.NIHL_all() 已废弃，请使用 calculate_all_NIHL() 代替"
        )
        return AuditoryDiagnose.calculate_all_NIHL(*args, **kwargs)

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
        if mean_key is None:
            mean_key = [3000, 4000, 6000]
            
        # 标准化性别
        sex_norm = "Male" if str(sex).startswith("M") or sex in ("男", "male") else "Female"
        age_boxed = AuditoryConstants.AGE_BOXING(age=age)
        percentrage_str = str(percentrage) + "pr"

        # 获取标准 PTA
        if standard == "Chinese":
            standard_PTA = AuditoryConstants.CHINESE_STANDARD_PTA_DICT.get(sex_norm, {}).get(age_boxed)
        else:
            standard_PTA = AuditoryConstants.ISO_1999_2013_STANDARD_PTA_DICT.get(sex_norm, {}).get(age_boxed)
        
        if standard_PTA is None:
            raise ValueError(f"Standard PTA data not found for sex: {sex_norm}, age: {age_boxed}")
            
        # 过滤频率
        standard_PTA = seq(standard_PTA.items()).filter(
            lambda x: int(x[0].split("Hz")[0]) in mean_key
        ).map(lambda x: (int(x[0].split("Hz")[0]), x[1])).dict()
        standard_PTA = seq(standard_PTA.items()).map(
            lambda x: (x[0], x[1].get(percentrage_str))).dict()

        # 计算预测值
        NIPTS_preds = []
        for freq in mean_key:
            params = AuditoryConstants.ISO_1999_2013_NIPTS_PRED_DICT.get(str(freq) + "Hz", {})
            u = params.get("u")
            v = params.get("v")
            L0 = params.get("L0")
                
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
            
        return np.mean(NIPTS_preds)

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
        if mean_key is None:
            mean_key = [3000, 4000, 6000]

        # 标准化输入参数
        age = 21 if age <= 20 else age
        age = 70 if age > 70 else age
        duration = 40 if duration > 40 else duration
        duration = age - 20 if age - duration < 20 else duration
        sex_str = "Male" if str(sex).startswith("M") or sex in ("男", "male") else "Female"
        
        # 获取插值参数
        p_idx, p_next_idx, p_ratio = AuditoryDiagnose._get_percentile_indices(percentrage)
        age_idx, age_next_idx, age_ratio = AuditoryDiagnose._get_age_interpolation_params(age)
        laeq_idx, laeq_next_idx, laeq_ratio = AuditoryDiagnose._get_laeq_interpolation_params(LAeq)
        duration_idx, duration_next_idx, duration_ratio = AuditoryDiagnose._get_duration_interpolation_params(duration)

        # 计算预测值
        nipts_predictions = []
        for freq in mean_key:
            nipts_pred = AuditoryDiagnose._calculate_nipts_for_frequency(
                freq=freq, duration=duration,
                laeq_idx=laeq_idx, laeq_next_idx=laeq_next_idx, laeq_ratio=laeq_ratio,
                duration_idx=duration_idx, duration_next_idx=duration_next_idx, duration_ratio=duration_ratio,
                p_idx=p_idx, p_next_idx=p_next_idx, p_ratio=p_ratio,
                age_idx=age_idx, age_next_idx=age_next_idx, age_ratio=age_ratio,
                age=age, sex=sex_str, LAeq=LAeq,
                extrapolation=extrapolation, NH_limit=NH_limit,
                percentrage=percentrage, mean_key=mean_key
            )
            if not np.isnan(nipts_pred):
                nipts_predictions.append(nipts_pred)

        if not nipts_predictions:
            raise ValueError("No valid predictions could be calculated")
            
        return np.mean(nipts_predictions)

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