"""
听力损失预测拟合函数

用于噪声诱导听力损失（NIL）的建模
"""
import numpy as np


class NILFunction():
    """噪声诱导听力损失拟合函数类"""
    
    @staticmethod
    def age_NIL(x, b0, b1, b2):
        """年龄 + 噪声暴露的线性拟合"""
        age, NIL = x
        return b0 + b1 * age + b2 * NIL

    @staticmethod
    def age_NIL_kurtosis(x, b0, b1, b2, b3):
        """年龄 + 噪声暴露 + 峰度的线性拟合"""
        age, NIL, kurtosis = x
        return b0 + b1 * age + b2 * NIL + b3 * kurtosis
