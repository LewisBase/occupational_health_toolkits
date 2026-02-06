"""
ISO 标准 NIPTS 预测器

实现 ISO 1999:2013 和 ISO 1999:2023 标准的 NIPTS 预测
"""
import numpy as np
from typing import List, Optional

from loguru import logger

from ohtk.diagnose_info.nipts_predictor.base import (
    BaseNIPTSPredictor,
    NIPTSPredictionResult
)
from ohtk.diagnose_info.nipts_predictor.factory import NIPTSPredictorFactory
from ohtk.constants.auditory_constants import AuditoryConstants


class ISO1999_2013Predictor(BaseNIPTSPredictor):
    """ISO 1999:2013 标准 NIPTS 预测器"""
    
    name = "iso1999_2013"
    version = "1.0.0"
    supported_ranges = {
        "LAeq": (70, 120),
        "age": (18, 70),
        "duration": (1, 40),
    }
    
    def predict(
        self,
        LAeq: float,
        age: int,
        sex: str,
        duration: float,
        percentrage: int = 50,
        mean_key: Optional[List[int]] = None,
        Hs: bool = False,
        standard: str = "Chinese",
        NH_limit: bool = True,
        **kwargs
    ) -> NIPTSPredictionResult:
        """基于 ISO 1999:2013 预测 NIPTS
        
        Args:
            LAeq: 等效连续A计权声压级 (dB)
            age: 年龄
            sex: 性别
            duration: 接噪工龄 (年)
            percentrage: 百分位数
            mean_key: 频率列表
            Hs: 是否考虑基底听力损失
            standard: 标准类型
            NH_limit: 是否限制非噪声相关听力损失
            
        Returns:
            NIPTSPredictionResult: 预测结果
        """
        from functional import seq
        
        if mean_key is None:
            mean_key = [3000, 4000, 6000]
        
        # 标准化性别
        sex_norm = self._normalize_sex(sex)
        age_boxed = AuditoryConstants.AGE_BOXING(age=age)
        percentrage_str = str(percentrage) + "pr"
        
        # 获取标准 PTA
        if standard == "Chinese":
            standard_PTA = AuditoryConstants.CHINESE_STANDARD_PTA_DICT.get(sex_norm, {}).get(age_boxed)
        else:
            standard_PTA = AuditoryConstants.ISO_1999_2013_STANDARD_PTA_DICT.get(sex_norm, {}).get(age_boxed)
        
        if standard_PTA is None:
            return NIPTSPredictionResult(
                value=float('nan'),
                method=self.name,
                metadata={"error": f"Standard PTA data not found for sex: {sex_norm}, age: {age_boxed}"}
            )
        
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
            return NIPTSPredictionResult(
                value=float('nan'),
                method=self.name,
                metadata={"error": "No valid predictions could be calculated"}
            )
        
        return NIPTSPredictionResult(
            value=float(np.mean(NIPTS_preds)),
            method=self.name,
            metadata={
                "standard": standard,
                "frequencies": mean_key,
                "percentrage": percentrage,
            }
        )


class ISO1999_2023Predictor(BaseNIPTSPredictor):
    """ISO 1999:2023 标准 NIPTS 预测器"""
    
    name = "iso1999_2023"
    version = "1.0.0"
    supported_ranges = {
        "LAeq": (70, 120),
        "age": (20, 70),
        "duration": (1, 40),
    }
    
    def predict(
        self,
        LAeq: float,
        age: int,
        sex: str,
        duration: float,
        percentrage: int = 50,
        mean_key: Optional[List[int]] = None,
        extrapolation: Optional[str] = None,
        NH_limit: bool = True,
        **kwargs
    ) -> NIPTSPredictionResult:
        """基于 ISO 1999:2023 预测 NIPTS
        
        Args:
            LAeq: 等效连续A计权声压级 (dB)
            age: 年龄
            sex: 性别
            duration: 接噪工龄 (年)
            percentrage: 百分位数
            mean_key: 频率列表
            extrapolation: 外推方法 ("ML" 或 "Linear")
            NH_limit: 是否限制非噪声相关听力损失
            
        Returns:
            NIPTSPredictionResult: 预测结果
        """
        if mean_key is None:
            mean_key = [3000, 4000, 6000]
        
        # 标准化输入参数
        age = 21 if age <= 20 else age
        age = 70 if age > 70 else age
        duration = 40 if duration > 40 else duration
        duration = age - 20 if age - duration < 20 else duration
        sex_str = self._normalize_sex(sex)
        
        # 获取插值参数
        p_idx, p_next_idx, p_ratio = self._get_percentile_indices(percentrage)
        age_idx, age_next_idx, age_ratio = self._get_age_interpolation_params(age)
        laeq_idx, laeq_next_idx, laeq_ratio = self._get_laeq_interpolation_params(LAeq)
        duration_idx, duration_next_idx, duration_ratio = self._get_duration_interpolation_params(duration)
        
        # 计算预测值
        nipts_predictions = []
        for freq in mean_key:
            nipts_pred = self._calculate_nipts_for_frequency(
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
            return NIPTSPredictionResult(
                value=float('nan'),
                method=self.name,
                metadata={"error": "No valid predictions could be calculated"}
            )
        
        return NIPTSPredictionResult(
            value=float(np.mean(nipts_predictions)),
            method=self.name,
            metadata={
                "frequencies": mean_key,
                "percentrage": percentrage,
                "extrapolation": extrapolation,
            }
        )
    
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
        A1 = min(A1, len(As) - 2)
        A2 = A1 + 1
        AR = (A0 - A1) / (A2 - A1)
        return A1, A2, AR
    
    def _get_laeq_interpolation_params(self, LAeq: float) -> tuple:
        """获取 LAeq 的插值参数"""
        ls = [70, 75, 80, 85, 90, 95, 100]
        L0 = (LAeq - 65) / 5
        L1 = int(L0) if LAeq < 100 else 6
        L1 = min(L1, len(ls) - 2)
        L2 = L1 + 1
        LR = (L0 - L1) / (L2 - L1)
        return L1, L2, LR
    
    def _get_duration_interpolation_params(self, duration: float) -> tuple:
        """获取工龄的插值参数"""
        D0 = duration / 10
        D1 = int(D0) if duration != 40 else 3
        D1 = D1 if duration >= 10 else 1
        D2 = D1 + 1
        DR = (D0 - D1) / (D2 - D1)
        return D1, D2, DR
    
    def _dict_query_nipts(self, laeq_idx: int, duration_idx: int, p_idx: int, freq: int) -> float:
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
    
    def _calculate_nipts_for_frequency(
        self,
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
            nipts_ldq = self._handle_extrapolation(
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
    
    def _handle_extrapolation(
        self,
        LAeq: float, age: int, duration: float, sex: str, freq: int,
        percentrage: int, mean_key: List[int],
        extrapolation: Optional[str], H: float, NH_limit: bool
    ) -> float:
        """处理外推情况"""
        if extrapolation == "ML":
            try:
                import pickle
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
                nipts_pred_95 = self.predict(
                    LAeq=95, age=age, sex=sex, duration=duration,
                    percentrage=percentrage, mean_key=mean_key,
                    extrapolation=None, NH_limit=NH_limit
                ).value
                nipts_pred_100 = self.predict(
                    LAeq=100, age=age, sex=sex, duration=duration,
                    percentrage=percentrage, mean_key=mean_key,
                    extrapolation=None, NH_limit=NH_limit
                ).value
                slope = (nipts_pred_100 - nipts_pred_95) / 5
                return nipts_pred_100 + slope * (LAeq - 100)
            except RecursionError:
                logger.error("Recursion error during extrapolation, returning NaN")
                return np.nan
        else:
            return np.nan


# 注册预测器
NIPTSPredictorFactory.register("iso1999_2013", ISO1999_2013Predictor)
NIPTSPredictorFactory.register("iso1999_2023", ISO1999_2023Predictor)
