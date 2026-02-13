# -*- coding: utf-8 -*-
"""
特征构建器模块

提供特征提取、编码和预处理功能
支持可配置的分类列和缺失值填充
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from loguru import logger
from typing import Dict, List, Tuple, Optional, Union


class FeatureBuilder:
    """特征构建器类"""
    
    def __init__(self, 
                 feature_columns: List[str] = None,
                 categorical_columns: List[str] = None,
                 fill_missing: bool = True,
                 scale_features: bool = True):
        """
        初始化特征构建器
        
        Args:
            feature_columns: 特征列名称列表，None则使用默认列表
            categorical_columns: 分类列名称列表，None则自动检测
            fill_missing: 是否填充缺失值
            scale_features: 是否标准化数值特征
        """
        self.feature_columns = feature_columns or [
            "sex", "age", "duration", "work_shop", "work_position", 
            "smoking", "year_of_smoking", "Leq", "LAeq", "LCeq",
            "kurtosis_arimean", "A_kurtosis_arimean", "C_kurtosis_arimean",
            "kurtosis_geomean", "A_kurtosis_geomean", "C_kurtosis_geomean",
            "max_Peak_SPL_dB"
        ]
        
        self.categorical_columns = categorical_columns  # None表示自动检测
        self.fill_missing = fill_missing
        self.do_scale_features = scale_features  # 重命名以避免与方法名冲突
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._detected_categorical = []  # 存储自动检测到的分类列
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取指定特征列
        
        Args:
            df: 原始DataFrame
            
        Returns:
            包含特征的DataFrame
        """
        logger.info("提取特征列...")
        
        # 确保必要的列存在
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"缺失特征列: {missing_features}")
        
        logger.info(f"可用特征列: {len(available_features)}/{len(self.feature_columns)}")
        
        # 提取特征（统一使用 staff_id）
        id_col = 'staff_id' if 'staff_id' in df.columns else 'worker_id'
        if id_col not in df.columns:
            logger.warning("数据中既没有 staff_id 也没有 worker_id，将不包含ID列")
            feature_df = df[available_features].copy()
        else:
            feature_df = df[[id_col] + available_features].copy()
        
        return feature_df
    
    def _auto_detect_categorical(self, df: pd.DataFrame) -> List[str]:
        """
        自动检测分类列
        
        检测规则：
        1. object类型列
        2. 唯一值数量少于10的数值列
        3. 排除worker_id
        
        Args:
            df: DataFrame
            
        Returns:
            分类列名称列表
        """
        cat_cols = []
        
        for col in df.columns:
            if col == 'worker_id':
                continue
                
            # object类型
            if df[col].dtype == 'object':
                cat_cols.append(col)
            # 唯一值数量少于10的数值列
            elif df[col].nunique() < 10:
                cat_cols.append(col)
        
        self._detected_categorical = cat_cols
        logger.info(f"自动检测到的分类列: {cat_cols}")
        return cat_cols
    
    def get_categorical_columns(self, df: pd.DataFrame = None) -> List[str]:
        """
        获取分类列列表
        
        Args:
            df: 可选的DataFrame，用于自动检测
            
        Returns:
            分类列名称列表
        """
        if self.categorical_columns is not None:
            return self.categorical_columns
        elif df is not None:
            return self._auto_detect_categorical(df)
        else:
            return self._detected_categorical
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        编码分类特征
        
        Args:
            df: 包含特征的DataFrame
            fit: 是否训练编码器
            
        Returns:
            编码后的DataFrame
        """
        logger.info("编码分类特征...")
        
        df_encoded = df.copy()
        
        # 获取分类列
        categorical_columns = self.get_categorical_columns(df)
        
        if not categorical_columns:
            logger.info("没有分类列需要编码")
            return df_encoded
        
        for col in categorical_columns:
            if col not in df_encoded.columns:
                continue
                
            if fit:
                le = LabelEncoder()
                mask = df_encoded[col].notna()
                if mask.any():
                    try:
                        encoded_values = le.fit_transform(df_encoded.loc[mask, col].astype(str))
                        df_encoded[col] = df_encoded[col].astype('object')
                        df_encoded.loc[mask, col] = encoded_values
                        df_encoded[col] = df_encoded[col].astype(int)
                        self.label_encoders[col] = le
                        logger.info(f"训练 {col} 编码器，类别数: {len(le.classes_)}")
                    except Exception as e:
                        logger.warning(f"编码 {col} 失败: {e}")
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    mask = df_encoded[col].notna()
                    if mask.any():
                        try:
                            # 处理未见过的类别
                            unique_vals = df_encoded.loc[mask, col].astype(str).unique()
                            known_classes = set(le.classes_)
                            
                            # 替换未知类别为已知类别中的第一个
                            mapped_vals = []
                            for val in df_encoded.loc[mask, col].astype(str):
                                if val in known_classes:
                                    mapped_vals.append(val)
                                else:
                                    mapped_vals.append(le.classes_[0])  # 使用第一个类别作为默认值
                            
                            encoded_values = le.transform(mapped_vals)
                            df_encoded[col] = df_encoded[col].astype('object')
                            df_encoded.loc[mask, col] = encoded_values
                            df_encoded[col] = df_encoded[col].astype(int)
                        except Exception as e:
                            logger.warning(f"转换 {col} 失败: {e}")
        
        return df_encoded
    
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: DataFrame
            
        Returns:
            处理后的DataFrame
        """
        if not self.fill_missing:
            logger.info("跳过缺失值填充（fill_missing=False）")
            return df.copy()
        
        logger.info("处理缺失值...")
        
        df_handled = df.copy()
        
        # 数值型特征用中位数填充
        numeric_columns = df_handled.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'worker_id']
        
        for col in numeric_columns:
            if df_handled[col].isna().any():
                median_val = df_handled[col].median()
                df_handled[col] = df_handled[col].fillna(median_val)
                logger.debug(f"  {col}: 用中位数 {median_val:.4f} 填充")
        
        # 分类特征用众数填充
        categorical_columns = self.get_categorical_columns(df)
        for col in categorical_columns:
            if col in df_handled.columns and df_handled[col].isna().any():
                mode_val = df_handled[col].mode()
                if len(mode_val) > 0:
                    df_handled[col] = df_handled[col].fillna(mode_val[0])
                    logger.debug(f"  {col}: 用众数 {mode_val[0]} 填充")
        
        return df_handled
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        标准化数值特征
        
        Args:
            df: DataFrame
            fit: 是否训练缩放器
            
        Returns:
            标准化后的DataFrame
        """
        if not self.do_scale_features:
            logger.info("跳过特征标准化（scale_features=False）")
            return df.copy()
        
        logger.info("标准化数值特征...")
        
        df_scaled = df.copy()
        
        # 识别数值型特征（排除分类列和ID列）
        categorical_columns = set(self.get_categorical_columns(df))
        numerical_columns = []
        
        for col in df_scaled.columns:
            if col == 'worker_id' or col in categorical_columns:
                continue
            if df_scaled[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                unique_vals = df_scaled[col].nunique()
                # 唯一值大于10或者是浮点型
                if unique_vals > 10 or df_scaled[col].dtype in ['float64', 'float32']:
                    numerical_columns.append(col)
        
        if numerical_columns:
            if fit:
                df_scaled[numerical_columns] = self.scaler.fit_transform(df_scaled[numerical_columns])
                self._is_fitted = True
                logger.info(f"训练标准化器，特征数: {len(numerical_columns)}")
            else:
                df_scaled[numerical_columns] = self.scaler.transform(df_scaled[numerical_columns])
        else:
            logger.info("没有数值特征需要标准化")
        
        return df_scaled
    
    def prepare_features(self, df: pd.DataFrame, target_col: str, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征矩阵和目标变量
        
        Args:
            df: 包含特征和目标的DataFrame
            target_col: 目标变量列名
            fit: 是否拟合转换器
            
        Returns:
            (特征矩阵X, 目标变量y)
        """
        logger.info(f"准备训练数据，目标变量: {target_col}")
        
        # 提取特征
        feature_df = self.extract_features(df)
        
        # 处理缺失值
        feature_df = self.handle_missing(feature_df)
        
        # 编码分类特征
        feature_df = self.encode_categorical(feature_df, fit=fit)
        
        # 标准化数值特征
        feature_df = self.scale_features(feature_df, fit=fit)
        
        # 分离特征和目标变量
        feature_cols = [col for col in feature_df.columns if col != 'worker_id']
        X = feature_df[feature_cols]
        
        # 确保所有特征都是数值类型
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
        
        # 获取目标变量
        if target_col in df.columns:
            y = df[target_col]
            # 移除目标变量为空的样本
            valid_mask = y.notna()
            X = X[valid_mask]
            y = y[valid_mask]
        else:
            raise ValueError(f"目标变量 {target_col} 不存在于数据中")
        
        logger.info(f"特征矩阵: {X.shape}, 目标变量: {y.shape}")
        
        return X, y
    
    def prepare_all_targets(self, df: pd.DataFrame, target_cols: List[str]) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        为所有目标变量准备训练数据
        
        Args:
            df: 包含特征和目标的DataFrame
            target_cols: 目标变量列名列表
            
        Returns:
            目标变量到(特征矩阵,目标变量)的映射
        """
        logger.info(f"为 {len(target_cols)} 个目标变量准备训练数据...")
        
        results = {}
        
        for i, target_col in enumerate(target_cols):
            try:
                # 第一个目标变量需要fit，后续的只需要transform
                fit = (i == 0)
                X, y = self.prepare_features(df, target_col, fit=fit)
                results[target_col] = (X, y)
                logger.info(f"✓ {target_col}: X{X.shape}, y{y.shape}")
            except Exception as e:
                logger.error(f"✗ {target_col} 准备失败: {e}")
        
        return results