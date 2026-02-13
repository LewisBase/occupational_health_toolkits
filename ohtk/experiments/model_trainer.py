# -*- coding: utf-8 -*-
"""
模型训练器模块

提供标准化的模型训练和交叉验证功能
支持回归和分类任务，支持多种模型
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from loguru import logger
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')


class BaseTrainer(ABC):
    """训练器抽象基类"""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        初始化训练器
        
        Args:
            n_folds: 折数
            random_state: 随机种子
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = {}
    
    @abstractmethod
    def cross_validate(self, model_class, X, y, **kwargs) -> Dict[str, Any]:
        """执行交叉验证，子类必须实现"""
        pass
    
    def save_model(self, model, filepath: Path):
        """
        保存模型
        
        Args:
            model: 模型对象
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"模型已保存到: {filepath}")


class CrossValidationTrainer(BaseTrainer):
    """交叉验证训练器，支持回归和分类任务"""
    
    def __init__(self, 
                 task_type: str = 'regression',
                 n_folds: int = 5, 
                 random_state: int = 42):
        """
        初始化交叉验证训练器
        
        Args:
            task_type: 任务类型 ('regression' 或 'classification')
            n_folds: 折数
            random_state: 随机种子
        """
        super().__init__(n_folds, random_state)
        self.task_type = task_type
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
    def _get_metrics_by_task(self) -> Dict[str, callable]:
        """
        根据任务类型返回评估指标函数
        
        Returns:
            指标函数字典
        """
        if self.task_type == 'regression':
            return {
                'mse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
                'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
                'r2': lambda y_true, y_pred: r2_score(y_true, y_pred)
            }
        else:  # classification
            return {
                'accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
                'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
                'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
                'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
            }
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        分割数据为训练集和测试集
        
        Args:
            X: 特征矩阵
            y: 目标变量
            test_size: 测试集比例
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        logger.info(f"分割数据: 训练集 {1-test_size:.0%} / 测试集 {test_size:.0%}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def cross_validate(self, model_class, X: pd.DataFrame, y: pd.Series, 
                      model_params: Dict = None, fit_params: Dict = None) -> Dict[str, Any]:
        """
        执行交叉验证
        
        Args:
            model_class: 模型类
            X: 特征矩阵
            y: 目标变量
            model_params: 模型参数
            fit_params: 拟合参数
            
        Returns:
            交叉验证结果字典
        """
        logger.info(f"开始 {self.n_folds} 折交叉验证...")
        
        model_params = model_params or {}
        fit_params = fit_params or {}
        
        # 获取评估指标
        metric_funcs = self._get_metrics_by_task()
        cv_scores = {name: [] for name in metric_funcs.keys()}
        fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X)):
            logger.info(f"训练第 {fold + 1}/{self.n_folds} 折...")
            
            # 分割数据
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 创建并训练模型
            model = model_class(**model_params)
            model.fit(X_fold_train, y_fold_train, **fit_params)
            
            # 预测和评估
            y_pred = model.predict(X_fold_val)
            
            # 计算指标
            for metric_name, metric_func in metric_funcs.items():
                try:
                    score = metric_func(y_fold_val, y_pred)
                    cv_scores[metric_name].append(score)
                except Exception as e:
                    logger.warning(f"计算 {metric_name} 失败: {e}")
                    cv_scores[metric_name].append(np.nan)
            
            fold_models.append(model)
            
            # 打印关键指标
            if self.task_type == 'regression':
                logger.info(f"  Fold {fold + 1}: RMSE={cv_scores['rmse'][-1]:.4f}, R²={cv_scores['r2'][-1]:.4f}")
            else:
                logger.info(f"  Fold {fold + 1}: Acc={cv_scores['accuracy'][-1]:.4f}, F1={cv_scores['f1'][-1]:.4f}")
        
        # 汇总结果
        results = {
            'fold_scores': cv_scores,
            'fold_models': fold_models
        }
        
        # 计算平均值和标准差
        for metric_name in metric_funcs.keys():
            results[f'cv_mean_{metric_name}'] = np.mean(cv_scores[metric_name])
            results[f'cv_std_{metric_name}'] = np.std(cv_scores[metric_name])
        
        # 选择最佳模型
        if self.task_type == 'regression':
            best_idx = np.argmin(cv_scores['rmse'])  # RMSE最小
            logger.info(f"交叉验证完成: RMSE={results['cv_mean_rmse']:.4f}±{results['cv_std_rmse']:.4f}")
        else:
            best_idx = np.argmax(cv_scores['accuracy'])  # Accuracy最大
            logger.info(f"交叉验证完成: Accuracy={results['cv_mean_accuracy']:.4f}±{results['cv_std_accuracy']:.4f}")
        
        results['best_model'] = fold_models[best_idx]
        
        return results
    
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        在测试集上评估模型
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            测试集评估结果
        """
        y_pred = model.predict(X_test)
        
        # 获取评估指标
        metric_funcs = self._get_metrics_by_task()
        results = {}
        
        for metric_name, metric_func in metric_funcs.items():
            try:
                results[f'test_{metric_name}'] = metric_func(y_test, y_pred)
            except Exception as e:
                logger.warning(f"计算 test_{metric_name} 失败: {e}")
                results[f'test_{metric_name}'] = np.nan
        
        # 打印关键指标
        if self.task_type == 'regression':
            logger.info(f"测试集性能: RMSE={results.get('test_rmse', 0):.4f}, R²={results.get('test_r2', 0):.4f}")
        else:
            logger.info(f"测试集性能: Accuracy={results.get('test_accuracy', 0):.4f}, F1={results.get('test_f1', 0):.4f}")
        
        return results


class ModelWrapper:
    """模型包装器，统一不同模型的接口"""
    
    def __init__(self, model_type: str = 'lightgbm', **kwargs):
        """
        初始化模型包装器
        
        Args:
            model_type: 模型类型 ('lightgbm', 'tabtransformer', etc.)
            **kwargs: 模型参数
        """
        self.model_type = model_type
        self.model = None
        self.params = kwargs
        
    def fit(self, X, y, **kwargs):
        """训练模型"""
        if self.model_type == 'lightgbm':
            self._fit_lightgbm(X, y, **kwargs)
        elif self.model_type == 'tabtransformer':
            self._fit_tabtransformer(X, y, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _fit_lightgbm(self, X, y, early_stopping_rounds=50, num_boost_round=1000):
        """训练LightGBM模型"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("需要安装 lightgbm: pip install lightgbm")
        
        # 分割训练验证集用于早停
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 默认参数
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        params.update(self.params)
        
        # 训练
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.log_evaluation(0),  # 不输出日志
                lgb.early_stopping(early_stopping_rounds, verbose=False)
            ]
        )
    
    def _fit_tabtransformer(self, X, y, epochs=100, batch_size=256):
        """训练TabTransformer模型"""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("需要安装 pytorch: pip install torch")
        
        try:
            # 尝试导入现有的TabTransformer
            from ohtk.modeling.transformers import TabTransformer
            logger.info("使用 ohtk.modeling.transformers.TabTransformer")
        except ImportError:
            logger.warning("未找到ohtk TabTransformer，使用简化实现")
            # 简化实现：使用MLP代替
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=epochs,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
            self.model.fit(X, y)
            return
        
        # 准备数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 模型参数
        n_features = X.shape[1]
        model_params = {
            'num_features': n_features,
            'dim': self.params.get('dim', 32),
            'depth': self.params.get('depth', 6),
            'heads': self.params.get('heads', 8),
            'dim_head': self.params.get('dim_head', 16),
            'mlp_hidden_mults': self.params.get('mlp_hidden_mults', (4, 2)),
            'output_dim': 1
        }
        
        model = TabTransformer(**model_params)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.get('learning_rate', 0.001))
        
        # 训练
        best_val_loss = float('inf')
        patience = self.params.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停: epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
        
        # 恢复最佳模型
        model.load_state_dict(best_model_state)
        self.model = model
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if self.model_type == 'lightgbm':
            return self.model.predict(X, num_iteration=self.model.best_iteration)
        elif self.model_type == 'tabtransformer':
            # 检查是否是sklearn模型
            if hasattr(self.model, 'predict'):
                if hasattr(self.model, 'predict') and 'sklearn' in str(type(self.model)):
                    return self.model.predict(X)
                else:
                    # PyTorch模型
                    import torch
                    self.model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
                        predictions = self.model(X_tensor)
                        return predictions.numpy().flatten()
        else:
            return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if self.model_type == 'lightgbm':
            importance = self.model.feature_importance(importance_type='gain')
            feature_names = self.model.feature_name()
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            logger.warning(f"{self.model_type} 不支持特征重要性")
            return pd.DataFrame()