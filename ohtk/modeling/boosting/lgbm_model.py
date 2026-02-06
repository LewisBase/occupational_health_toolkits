# -*- coding: utf-8 -*-
"""
LightGBM 预测模型

用于构建NIHL预测的梯度提升模型，
支持特征工程、SHAP解释和模型保存。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import List, Optional, Dict, Any, Union


class LGBMNIHLPredictor:
    """
    基于LightGBM的NIHL预测模型
    
    支持纵向数据的特征工程，包括滞后特征、
    交互特征等，并提供SHAP解释能力。
    
    Attributes:
        outcome: 结果变量名
        exposure: 暴露变量名
        base_features: 基础特征列表
        group_var: 个体标识变量
    """
    
    def __init__(
        self,
        outcome: str = "NIHL346",
        exposure: str = "LAeq",
        base_features: Optional[List[str]] = None,
        group_var: str = "worker_id"
    ):
        """
        初始化LightGBM预测器
        
        Args:
            outcome: 结果变量名
            exposure: 暴露变量名
            base_features: 基础特征列表
            group_var: 个体标识变量
        """
        self.outcome = outcome
        self.exposure = exposure
        self.base_features = base_features or [
            "check_order", "age", "sex", "days_since_first"
        ]
        self.group_var = group_var
        self.model = None
        self.feature_names = None
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        add_interactions: bool = True,
        add_polynomial: bool = True,
        add_lags: bool = True,
        lag_features: Optional[List[str]] = None,
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        特征工程
        
        Args:
            df: 原始数据框
            add_interactions: 是否添加交互特征
            add_polynomial: 是否添加多项式特征
            add_lags: 是否添加滞后特征
            lag_features: 需要滞后的特征
            lags: 滞后阶数列表
        
        Returns:
            添加了工程特征的数据框
        """
        df = df.copy()
        
        # 确保暴露变量名称一致
        if self.exposure not in df.columns and "LEX_8h_LEX_40h_median" in df.columns:
            df[self.exposure] = df["LEX_8h_LEX_40h_median"]
        
        # 多项式特征
        if add_polynomial:
            if "check_order" in df.columns:
                df["check_order_sq"] = df["check_order"] ** 2
            if "days_since_first" in df.columns:
                df["days_since_first_sq"] = df["days_since_first"] ** 2
        
        # 交互特征
        if add_interactions and self.exposure in df.columns:
            if "age" in df.columns:
                df[f"age_{self.exposure}_interaction"] = (
                    df["age"] * df[self.exposure]
                )
            if "sex" in df.columns:
                df[f"sex_{self.exposure}_interaction"] = (
                    df["sex"] * df[self.exposure]
                )
            if "check_order" in df.columns:
                if "age" in df.columns:
                    df["age_exam_interaction"] = df["age"] * df["check_order"]
                if "sex" in df.columns:
                    df["sex_exam_interaction"] = df["sex"] * df["check_order"]
            if "days_since_first" in df.columns:
                if "age" in df.columns:
                    df["age_day_interaction"] = df["age"] * df["days_since_first"]
                if "sex" in df.columns:
                    df["sex_day_interaction"] = df["sex"] * df["days_since_first"]
        
        # 滞后特征
        if add_lags and self.group_var in df.columns:
            lag_features = lag_features or [self.outcome, self.exposure]
            lags = lags or [1, 2]
            
            df = df.groupby(self.group_var, group_keys=False).apply(
                lambda x: self._add_lagged_features(x, lag_features, lags)
            )
            df = df.reset_index(drop=True)
        
        return df
    
    def _add_lagged_features(
        self,
        group: pd.DataFrame,
        cols: List[str],
        lags: List[int]
    ) -> pd.DataFrame:
        """为每个个体添加滞后特征"""
        for col in cols:
            if col in group.columns:
                for lag in lags:
                    group[f"{col}_lag{lag}"] = group[col].shift(lag)
        return group
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """获取特征列"""
        # 基础特征
        features = [f for f in self.base_features if f in df.columns]
        features.append(self.exposure)
        
        # 多项式特征
        poly_features = ["check_order_sq", "days_since_first_sq"]
        features.extend([f for f in poly_features if f in df.columns])
        
        # 交互特征
        interaction_features = [
            f"age_{self.exposure}_interaction",
            f"sex_{self.exposure}_interaction",
            "age_exam_interaction",
            "sex_exam_interaction",
            "age_day_interaction",
            "sex_day_interaction",
        ]
        features.extend([f for f in interaction_features if f in df.columns])
        
        # 滞后特征
        lag_features = [
            f"{self.outcome}_lag1", f"{self.outcome}_lag2",
            f"{self.exposure}_lag1", f"{self.exposure}_lag2",
        ]
        features.extend([f for f in lag_features if f in df.columns])
        
        return list(set(features))  # 去重
    
    def fit(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            df: 已进行特征工程的数据框
            test_size: 测试集比例
            params: LightGBM参数
            num_boost_round: 最大迭代次数
            early_stopping_rounds: 早停轮数
        
        Returns:
            训练结果字典
        """
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
        except ImportError as e:
            logger.error("需要安装 lightgbm 和 scikit-learn")
            raise ImportError("lightgbm or sklearn not installed") from e
        
        # 删除含有缺失值的行（主要是滞后产生的）
        df = df.dropna()
        
        # 获取特征列
        self.feature_names = self._get_feature_columns(df)
        
        logger.info(f"使用特征: {self.feature_names}")
        
        X = df[self.feature_names]
        y = df[self.outcome]
        
        # 按个体划分训练测试集（避免数据泄露）
        if self.group_var in df.columns:
            unique_subjects = df[self.group_var].unique()
            train_subjects, test_subjects = train_test_split(
                unique_subjects, test_size=test_size, random_state=42
            )
            
            train_mask = df[self.group_var].isin(train_subjects)
            X_train, X_test = X[train_mask], X[~train_mask]
            y_train, y_test = y[train_mask], y[~train_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # 默认参数
        default_params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "num_threads": 4,
        }
        
        if params:
            default_params.update(params)
        
        # 训练
        callbacks = [lgb.log_evaluation(100)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[test_data],
            callbacks=callbacks
        )
        
        # 评估
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # 特征重要性
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importance(importance_type="gain")
        }).sort_values("importance", ascending=False)
        
        logger.info(f"测试集 RMSE: {rmse:.4f}")
        logger.info(f"测试集 R²: {r2:.4f}")
        logger.info(f"最佳迭代: {self.model.best_iteration}")
        
        return {
            "model": self.model,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "best_iteration": self.model.best_iteration,
            "feature_importance": importance_df,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        预测
        
        Args:
            df: 数据框（需要已进行特征工程）
        
        Returns:
            预测值数组
        """
        if self.model is None:
            raise RuntimeError("请先训练模型")
        
        X = df[self.feature_names]
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def explain(
        self,
        df: pd.DataFrame,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        SHAP解释
        
        Args:
            df: 数据框
            sample_size: 用于计算SHAP值的样本数
        
        Returns:
            SHAP分析结果
        """
        try:
            import shap
        except ImportError:
            logger.error("需要安装 shap: pip install shap")
            raise ImportError("shap not installed")
        
        if self.model is None:
            raise RuntimeError("请先训练模型")
        
        X = df[self.feature_names]
        if sample_size and len(X) > sample_size:
            X = X.sample(sample_size, random_state=42)
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        # 计算特征重要性（基于SHAP）
        shap_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": np.abs(shap_values).mean(axis=0)
        }).sort_values("importance", ascending=False)
        
        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "expected_value": explainer.expected_value,
            "shap_importance": shap_importance,
            "X": X,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("没有可保存的模型")
        
        path = Path(path)
        self.model.save_model(str(path))
        logger.info(f"模型已保存到: {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """加载模型"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed")
        
        path = Path(path)
        self.model = lgb.Booster(model_file=str(path))
        logger.info(f"模型已从 {path} 加载")


def train_lgbm_nihl(
    df: pd.DataFrame,
    outcome: str = "NIHL346",
    exposure: str = "LAeq",
    **kwargs
) -> Dict[str, Any]:
    """
    LightGBM NIHL预测的便捷函数
    
    Args:
        df: 原始数据框
        outcome: 结果变量名
        exposure: 暴露变量名
        **kwargs: 传递给fit()的其他参数
    
    Returns:
        训练结果
    """
    predictor = LGBMNIHLPredictor(outcome=outcome, exposure=exposure)
    
    # 特征工程
    df_engineered = predictor.engineer_features(df)
    
    # 训练
    return predictor.fit(df_engineered, **kwargs)
