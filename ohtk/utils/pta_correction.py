# -*- coding: utf-8 -*-
"""
PTA (Pure Tone Audiometry) Correction Module

Provides age-based correction for hearing threshold values and NIHL calculation.
Based on Chinese occupational health standards.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, List

# Age-based PTA correction table (Chinese standard)
PTA_ADJUST_DICT = {
    "0-19": {
        "500": {"M": 0, "F": 0},
        "1000": {"M": 0, "F": 0},
        "2000": {"M": 0, "F": 0},
        "3000": {"M": 0, "F": 0},
        "4000": {"M": 0, "F": 0},
        "6000": {"M": 0, "F": 0},
    },
    "20-29": {
        "500": {"M": 0, "F": 0},
        "1000": {"M": 0, "F": 0},
        "2000": {"M": 0, "F": 0},
        "3000": {"M": 0, "F": 0},
        "4000": {"M": 0, "F": 0},
        "6000": {"M": 0, "F": 0},
    },
    "30-39": {
        "500": {"M": 1, "F": 1},
        "1000": {"M": 1, "F": 1},
        "2000": {"M": 1, "F": 1},
        "3000": {"M": 2, "F": 1},
        "4000": {"M": 2, "F": 1},
        "6000": {"M": 3, "F": 2},
    },
    "40-49": {
        "500": {"M": 2, "F": 2},
        "1000": {"M": 2, "F": 2},
        "2000": {"M": 3, "F": 3},
        "3000": {"M": 6, "F": 4},
        "4000": {"M": 8, "F": 4},
        "6000": {"M": 9, "F": 6},
    },
    "50-59": {
        "500": {"M": 4, "F": 4},
        "1000": {"M": 4, "F": 4},
        "2000": {"M": 7, "F": 6},
        "3000": {"M": 12, "F": 8},
        "4000": {"M": 16, "F": 9},
        "6000": {"M": 18, "F": 12},
    },
    "60-69": {
        "500": {"M": 6, "F": 6},
        "1000": {"M": 7, "F": 7},
        "2000": {"M": 12, "F": 11},
        "3000": {"M": 20, "F": 13},
        "4000": {"M": 28, "F": 16},
        "6000": {"M": 32, "F": 21},
    },
    "70-": {
        "500": {"M": 9, "F": 9},
        "1000": {"M": 11, "F": 11},
        "2000": {"M": 19, "F": 16},
        "3000": {"M": 31, "F": 20},
        "4000": {"M": 43, "F": 24},
        "6000": {"M": 49, "F": 32},
    },
}

# NIHL frequency configurations
NIHL_FREQ_CONFIG = {
    "1234": [1000, 2000, 3000, 4000],  # Speech frequencies
    "346": [3000, 4000, 6000],          # High frequencies
    "512": [500, 1000, 2000],           # Low frequencies (optional)
}


def get_age_segment(age: int) -> str:
    """
    Get age segment label based on age.
    
    Args:
        age: Age in years
        
    Returns:
        Age segment string (e.g., "30-39", "40-49")
    """
    if age < 20:
        return "0-19"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    else:
        return "70-"


def normalize_sex(sex: str) -> str:
    """
    Normalize sex value to 'M' or 'F'.
    
    Args:
        sex: Sex value in various formats
        
    Returns:
        'M' for male, 'F' for female
    """
    if sex in ("Male", "男", "M", "m", "male", "1", 1):
        return "M"
    return "F"


def get_correction_value(age: int, sex: str, freq: int) -> float:
    """
    Get age correction value for a specific frequency.
    
    Args:
        age: Age in years
        sex: Sex ('M' or 'F', or other formats that will be normalized)
        freq: Frequency in Hz (e.g., 1000, 2000, 3000, 4000, 6000)
        
    Returns:
        Correction value in dB
    """
    age_seg = get_age_segment(age)
    sex_norm = normalize_sex(sex)
    freq_str = str(freq)
    
    if age_seg not in PTA_ADJUST_DICT:
        return 0.0
    if freq_str not in PTA_ADJUST_DICT[age_seg]:
        return 0.0
    if sex_norm not in PTA_ADJUST_DICT[age_seg][freq_str]:
        return 0.0
    
    return float(PTA_ADJUST_DICT[age_seg][freq_str][sex_norm])


def correct_pta_value(raw_value: float, age: int, sex: str, freq: int) -> float:
    """
    Apply age correction to a single PTA value.
    
    Args:
        raw_value: Raw hearing threshold value in dB
        age: Age in years
        sex: Sex ('M' or 'F')
        freq: Frequency in Hz
        
    Returns:
        Corrected hearing threshold value in dB
    """
    if np.isnan(raw_value):
        return np.nan
    correction = get_correction_value(age, sex, freq)
    return raw_value - correction


def calculate_nihl(
    ear_data: Dict[str, float],
    freq_key: str = "346",
    age: Optional[int] = None,
    sex: Optional[str] = None,
    apply_correction: bool = False
) -> float:
    """
    Calculate NIHL (Noise-Induced Hearing Loss) indicator.
    
    NIHL is calculated as the average of hearing thresholds across specified frequencies
    for both ears.
    
    Args:
        ear_data: Dictionary with keys like 'left_ear_1000', 'right_ear_4000', etc.
                  Values are hearing thresholds in dB.
        freq_key: Frequency configuration key:
                  - "1234": 1000, 2000, 3000, 4000 Hz (speech frequencies)
                  - "346": 3000, 4000, 6000 Hz (high frequencies)
        age: Age in years (required if apply_correction=True)
        sex: Sex 'M' or 'F' (required if apply_correction=True)
        apply_correction: Whether to apply age correction
        
    Returns:
        NIHL value in dB (average across frequencies and ears)
        
    Example:
        >>> ear_data = {
        ...     'left_ear_3000': 25.0, 'right_ear_3000': 30.0,
        ...     'left_ear_4000': 35.0, 'right_ear_4000': 40.0,
        ...     'left_ear_6000': 45.0, 'right_ear_6000': 50.0,
        ... }
        >>> calculate_nihl(ear_data, freq_key="346")
        37.5
    """
    if freq_key not in NIHL_FREQ_CONFIG:
        raise ValueError(f"Unknown freq_key: {freq_key}. Available: {list(NIHL_FREQ_CONFIG.keys())}")
    
    frequencies = NIHL_FREQ_CONFIG[freq_key]
    values = []
    
    for freq in frequencies:
        for side in ["left", "right"]:
            # Try different key formats
            key_formats = [
                f"{side}_ear_{freq}",
                f"{side}_ear_{freq}_corr",
                f"{side}_{freq}",
                freq if side == "left" else None,  # Skip if not found
            ]
            
            value = None
            for key in key_formats:
                if key and key in ear_data:
                    value = ear_data[key]
                    break
            
            if value is not None and not np.isnan(value):
                if apply_correction and age is not None and sex is not None:
                    value = correct_pta_value(value, age, sex, freq)
                values.append(value)
    
    if not values:
        return np.nan
    
    return float(np.mean(values))


def calculate_all_nihl(
    ear_data: Dict[str, float],
    age: Optional[int] = None,
    sex: Optional[str] = None,
    apply_correction: bool = False
) -> Dict[str, float]:
    """
    Calculate all NIHL indicators.
    
    Args:
        ear_data: Dictionary with hearing threshold data
        age: Age in years (for correction)
        sex: Sex 'M' or 'F' (for correction)
        apply_correction: Whether to apply age correction
        
    Returns:
        Dictionary with NIHL values for each configuration:
        {"1234": float, "346": float}
    """
    result = {}
    for freq_key in ["1234", "346"]:
        result[freq_key] = calculate_nihl(
            ear_data, freq_key, age, sex, apply_correction
        )
    return result


def classify_hearing_loss(nihl_value: float) -> str:
    """
    Classify hearing loss severity based on NIHL value.
    
    Args:
        nihl_value: NIHL value in dB
        
    Returns:
        Classification string: "正常", "轻度", "中度", "重度"
    """
    if np.isnan(nihl_value):
        return "未知"
    if nihl_value <= 25:
        return "正常"
    elif nihl_value <= 40:
        return "轻度"
    elif nihl_value <= 55:
        return "中度"
    else:
        return "重度"


def convert_nihl_to_labels(
    nihl_values: Union[pd.Series, np.ndarray, List[float]],
    encoding: str = "categorical"
) -> Union[pd.Series, np.ndarray]:
    """
    将 NIHL 连续值批量转换为分类标签
    
    分类规则（基于 GBZ 49-2014）:
    - 0-25 dB: 正常 (0)
    - 25-40 dB: 轻度听力损失 (1)
    - 40-55 dB: 中度听力损失 (2)
    - 55+ dB: 重度听力损失 (3)
    
    Args:
        nihl_values: NIHL 连续值（Series, array, list）
        encoding: 编码方式
            - "categorical": 返回中文标签（"正常", "轻度", "中度", "重度"）
            - "numeric": 返回数字标签（0, 1, 2, 3）
            - "onehot": 返回 one-hot 编码矩阵 (4 列)
    
    Returns:
        编码后的标签
        - categorical: pd.Series (dtype=category)
        - numeric: np.ndarray (dtype=int)
        - onehot: np.ndarray (shape: n x 4)
    
    示例:
        >>> convert_nihl_to_labels([20, 30, 45, 60])
        0    正常
        1    轻度
        2    中度
        3    重度
        dtype: category
        
        >>> convert_nihl_to_labels([20, 30, 45, 60], encoding="numeric")
        array([0, 1, 2, 3])
    """
    # 转换为 numpy array
    if isinstance(nihl_values, pd.Series):
        arr = nihl_values.values
    elif isinstance(nihl_values, list):
        arr = np.array(nihl_values)
    else:
        arr = nihl_values
    
    # 分类阈值
    bins = [0, 25, 40, 55, np.inf]
    labels_categorical = ["正常", "轻度", "中度", "重度"]
    
    if encoding == "categorical":
        # 使用 pd.cut 进行分箱（返回 category 类型）
        result = pd.cut(
            arr, 
            bins=bins, 
            labels=labels_categorical, 
            right=True,
            include_lowest=True
        )
        return pd.Series(result)
    
    elif encoding == "numeric":
        # 使用 np.digitize 进行分箱（返回 0-3 的整数）
        # digitize 返回的是落入哪个 bin，需要减 1 变成 0-based
        result = np.digitize(arr, bins=[25, 40, 55]) 
        # 处理 NaN
        result = np.where(np.isnan(arr), -1, result).astype(int)
        return result
    
    elif encoding == "onehot":
        # 先获取 numeric 编码
        numeric = np.digitize(arr, bins=[25, 40, 55])
        # 创建 one-hot 矩阵
        n_samples = len(arr)
        onehot = np.zeros((n_samples, 4), dtype=int)
        for i, val in enumerate(numeric):
            if not np.isnan(arr[i]) and 0 <= val < 4:
                onehot[i, val] = 1
        return onehot
    
    else:
        raise ValueError(f"不支持的编码方式: {encoding}，可选: categorical, numeric, onehot")
