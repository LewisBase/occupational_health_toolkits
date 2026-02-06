# -*- coding: utf-8 -*-
"""
队列数据处理工具

用于纵向/队列研究数据的预处理和转换。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Optional, Union, Dict


class QueueDataProcessor:
    """
    队列数据处理器
    
    用于处理职业健康纵向队列数据，
    包括数据清洗、时间特征计算等。
    
    Attributes:
        group_var: 个体标识变量
        time_var: 时间变量（检查日期）
        required_columns: 必需的列
    """
    
    def __init__(
        self,
        group_var: str = "worker_id",
        time_var: str = "creation_date",
        required_columns: Optional[List[str]] = None
    ):
        """
        初始化队列数据处理器
        
        Args:
            group_var: 个体标识变量名
            time_var: 时间变量名
            required_columns: 必需的列
        """
        self.group_var = group_var
        self.time_var = time_var
        self.required_columns = required_columns or [
            "NIHL1234", "NIHL346", "worker_id",
            "creation_date", "sex", "age", "LEX_8h_LEX_40h_median"
        ]
    
    def load_and_process(
        self,
        input_path: Union[str, Path],
        filename: str = "lex_aggregated_by_report_median_queue_data.csv"
    ) -> Optional[pd.DataFrame]:
        """
        加载并处理队列数据
        
        Args:
            input_path: 输入路径
            filename: 文件名
        
        Returns:
            处理后的DataFrame，如果失败则返回None
        """
        input_path = Path(input_path)
        file_path = input_path / filename
        
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return None
        
        # 加载数据
        queue_df = pd.read_csv(file_path)
        logger.info(f"加载数据: {len(queue_df)} 行")
        
        # 检查必要列
        missing_columns = self._check_required_columns(queue_df)
        if missing_columns:
            logger.warning(f"缺失列: {missing_columns}")
            return None
        
        # 处理数据
        return self.process(queue_df)
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理队列数据
        
        Args:
            df: 原始数据框
        
        Returns:
            处理后的数据框
        """
        logger.info("处理队列数据...")
        
        # 排序
        df = df.sort_values(by=[self.group_var, self.time_var]).copy()
        
        # 计算检查次序
        df["check_order"] = df.groupby(self.group_var).cumcount() + 1
        
        # 分布信息
        check_dist = df["check_order"].value_counts().sort_index()
        logger.info(f"检查次序分布:\n{check_dist.head(10)}")
        
        # 计算时间特征
        df = self._add_time_features(df)
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        # 清理数据
        df = self._clean_data(df)
        
        return df
    
    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        """检查必需的列"""
        return [col for col in self.required_columns if col not in df.columns]
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        # 转换日期
        df["check_date"] = pd.to_datetime(df[self.time_var])
        
        # 距离首次检查的天数
        df["days_since_first"] = df.groupby(self.group_var)["check_date"].transform(
            lambda x: (x - x.iloc[0]).dt.days
        )
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        # 检查缺失值
        model_columns = [
            col for col in self.required_columns 
            if col in df.columns
        ]
        model_columns.extend(["check_order"])
        
        missing_check = df[model_columns].isnull().sum()
        if missing_check.any():
            logger.info(f"缺失值统计:\n{missing_check[missing_check > 0]}")
        
        # 删除缺失值
        original_len = len(df)
        df = df.dropna(subset=model_columns).copy()
        df = df.reset_index(drop=True)
        
        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"删除 {removed} 行含缺失值的数据")
        
        logger.info(f"清理后数据量: {len(df)} 行")
        
        return df
    
    def add_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2]
    ) -> pd.DataFrame:
        """
        添加滞后特征
        
        Args:
            df: 数据框
            columns: 需要滞后的列
            lags: 滞后阶数
        
        Returns:
            添加滞后特征后的数据框
        """
        def _add_lags(group):
            for col in columns:
                if col in group.columns:
                    for lag in lags:
                        group[f"{col}_lag{lag}"] = group[col].shift(lag)
            return group
        
        df = df.groupby(self.group_var, group_keys=False).apply(_add_lags)
        df = df.reset_index(drop=True)
        
        return df
    
    def get_baseline_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取基线数据（每个个体的首次检查）
        
        Args:
            df: 处理后的数据框
        
        Returns:
            基线数据框
        """
        baseline = df[df["check_order"] == 1].copy()
        logger.info(f"基线数据: {len(baseline)} 个个体")
        return baseline
    
    def get_followup_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        获取随访数据（排除首次检查）
        
        Args:
            df: 处理后的数据框
        
        Returns:
            随访数据框
        """
        followup = df[df["check_order"] > 1].copy()
        logger.info(f"随访数据: {len(followup)} 条记录")
        return followup
    
    def summary(self, df: pd.DataFrame) -> Dict:
        """
        数据摘要
        
        Args:
            df: 数据框
        
        Returns:
            摘要统计字典
        """
        n_subjects = df[self.group_var].nunique()
        n_records = len(df)
        check_order_stats = df.groupby(self.group_var)["check_order"].max()
        
        summary = {
            "n_subjects": n_subjects,
            "n_records": n_records,
            "avg_visits": n_records / n_subjects,
            "max_visits": check_order_stats.max(),
            "visit_distribution": check_order_stats.value_counts().to_dict(),
        }
        
        logger.info(f"数据摘要: {n_subjects} 个体, {n_records} 记录, "
                   f"平均 {summary['avg_visits']:.1f} 次访问")
        
        return summary


def load_queue_data(
    input_path: Union[str, Path],
    filename: str = "lex_aggregated_by_report_median_queue_data.csv",
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    加载队列数据的便捷函数
    
    Args:
        input_path: 输入路径
        filename: 文件名
        **kwargs: 传递给QueueDataProcessor的参数
    
    Returns:
        处理后的DataFrame
    """
    processor = QueueDataProcessor(**kwargs)
    return processor.load_and_process(input_path, filename)
