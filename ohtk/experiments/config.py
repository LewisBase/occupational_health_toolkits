# -*- coding: utf-8 -*-
"""
实验配置管理模块

提供YAML配置文件的读取、验证和管理功能
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ExperimentConfig:
    """实验配置管理类"""
    
    DEFAULT_CONFIG = {
        'experiment_name': 'experiment',
        'data': {
            'file_path': 'examples/data/All_Chinese_worker_exposure_data_0401.xlsx',
            'use_staffinfo': True
        },
        'features': {
            'feature_columns': None,  # None表示使用默认列表
            'categorical_columns': None,  # None表示自动检测
            'fill_missing': True,
            'scale_features': True
        },
        'models': [
            {
                'name': 'LightGBM',
                'type': 'lightgbm',
                'params': {
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8
                }
            }
        ],
        'targets': ['NIHL_1234', 'NIHL_346'],
        'training': {
            'n_folds': 5,
            'test_size': 0.2,
            'random_state': 42,
            'task_type': 'regression'
        },
        'evaluation': {
            'metrics': ['rmse', 'mae', 'mape', 'r2'],
            'compare_with_iso': False
        },
        'output': {
            'dir': 'results',
            'save_models': True,
            'generate_report': True
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化配置
        
        Args:
            config: 配置字典，None则使用默认配置
        """
        if config is None:
            self.config = self.DEFAULT_CONFIG.copy()
        else:
            # 合并默认配置和用户配置
            self.config = self._merge_configs(self.DEFAULT_CONFIG, config)
        
        # 验证配置
        self._validate_config()
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """
        递归合并配置字典
        
        Args:
            default: 默认配置
            custom: 用户自定义配置
            
        Returns:
            合并后的配置
        """
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self):
        """验证配置的有效性"""
        # 验证必需的键
        required_keys = ['experiment_name', 'data', 'models', 'targets', 'training', 'output']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"配置文件缺少必需的键: {key}")
        
        # 验证数据路径
        if 'file_path' not in self.config['data']:
            raise ValueError("配置文件缺少数据文件路径: data.file_path")
        
        # 验证模型配置
        if not self.config['models']:
            raise ValueError("至少需要配置一个模型")
        
        for model in self.config['models']:
            if 'name' not in model or 'type' not in model:
                raise ValueError("模型配置必须包含 name 和 type")
        
        # 验证目标变量
        if not self.config['targets']:
            raise ValueError("至少需要配置一个目标变量")
        
        logger.info("配置验证通过")
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'ExperimentConfig':
        """
        从YAML文件加载配置
        
        Args:
            yaml_path: YAML文件路径
            
        Returns:
            ExperimentConfig实例
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            logger.warning(f"配置文件不存在: {yaml_path}，使用默认配置")
            return cls()
        
        logger.info(f"从YAML文件加载配置: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return cls(config)
    
    def to_yaml(self, yaml_path: Path):
        """
        保存配置到YAML文件
        
        Args:
            yaml_path: YAML文件路径
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        logger.info(f"配置已保存到: {yaml_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键（支持点分隔的多级键，如 'data.file_path'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置项
        
        Args:
            key: 配置键（支持点分隔的多级键）
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self.config[key] = value
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"ExperimentConfig(experiment_name='{self.config['experiment_name']}')"
