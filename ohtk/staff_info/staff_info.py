import pandas as pd
import numpy as np
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Union, Optional, List, Any, ClassVar
from loguru import logger

from ohtk.staff_info.staff_basic_info import StaffBasicInfo
from ohtk.staff_info.staff_health_info import StaffHealthInfo
from ohtk.staff_info.staff_occhaz_info import StaffOccHazInfo


class StaffInfo(BaseModel):
    """
    职业健康检查员工信息模型
    
    支持时间序列数据，自动计算 check_order 和 days_since_first。
    
    数据加载方式：
    1. 直接传入字典: StaffInfo(staff_id="001", creation_date=[...], ...)
    2. 从 DataFrame 加载: StaffInfo.load_from_dataframe(df, worker_id)
    3. 批量加载: StaffInfo.load_batch_from_dataframe(df)
    
    查看支持的字段：
        StaffInfo.get_expected_fields()
        StaffInfo.help()
    """
    
    staff_id: Union[str, int]
    staff_name: str = ""
    staff_sex: Optional[str] = None
    staff_age: Optional[int] = None
    staff_duration: Optional[float] = None
    
    # Use datetime as key for time-series support
    staff_basic_info: Dict[datetime, StaffBasicInfo] = {}
    staff_health_info: Dict[datetime, StaffHealthInfo] = {}
    staff_occhaz_info: Dict[datetime, StaffOccHazInfo] = {}
    
    # Time-series features
    record_dates: List[datetime] = []
    check_order: Dict[datetime, int] = {}           # Check sequence number
    days_since_first: Dict[datetime, int] = {}      # Days since first check
    
    conflict_attr: list = []
    current_year: int = datetime.now().year
    
    # ========================================
    # 字段映射配置
    # ========================================
    
    # 基础信息字段 -> StaffBasicInfo 属性映射
    BASIC_INFO_FIELDS: ClassVar[Dict[str, str]] = {
        "name": "name",
        "factory_name": "factory_name",
        "work_shop": "work_shop",
        "work_position": "work_position",
        "smoking": "smoking",
        "year_of_smoking": "year_of_smoking",
        "cigaretee_per_day": "cigarette_per_day",  # 注意拼写映射
        "cigarette_per_day": "cigarette_per_day",
        "occupational_clinic_class": "occupational_clinic_class",
        "creation_province": "creation_province",
        "creation_city": "creation_city",
        "industry_category": "industry_category",
        "employment_unit_unified_social_credit_code": "employment_unit_code",
        "employment_unit_code": "employment_unit_code",
        "examination_type_code": "examination_type_code",
    }
    
    # 健康信息字段
    HEALTH_INFO_FIELDS: ClassVar[Dict[str, str]] = {
        "NIHL1234": "NIHL.1234",
        "NIHL346": "NIHL.346",
        "norm_hearing_loss": "norm_hearing_loss",
        "high_hearing_loss": "high_hearing_loss",
        "auditory_detection": "detection",
        "auditory_diagnose": "diagnose",
    }
    
    # 职业危害字段
    OCCHAZ_INFO_FIELDS: ClassVar[Dict[str, str]] = {
        "LEX_8h_LEX_40h_median": "LAeq",
        "LAeq": "LAeq",
        "noise_hazard_info": "noise_hazard_info",
    }
    
    # StaffInfo 直接属性字段
    STAFF_DIRECT_FIELDS: ClassVar[Dict[str, str]] = {
        "sex": "staff_sex",
        "age": "staff_age",
        "duration": "staff_duration",
    }

    def __init__(self, **data):
        # Extract check_order and days_since_first before Pydantic validation
        _check_order_raw = data.pop("check_order", None)
        _days_since_first_raw = data.pop("days_since_first", None)
        
        super().__init__(**data)
        
        # Store raw values for use in build methods
        data["check_order"] = _check_order_raw
        data["days_since_first"] = _days_since_first_raw
        
        # Build from dates (the only supported format now)
        if "creation_date" in data or self._has_datetime_data(data):
            self._build_from_dates(**data)
        else:
            # Try to auto-detect and build
            self._build_auto(**data)

    def _has_datetime_data(self, data: dict) -> bool:
        """Check if data contains datetime format."""
        for key in ["creation_date", "check_date"]:
            if key in data and data[key]:
                return True
        return False

    def _build_auto(self, **data):
        """Auto-detect data format and build."""
        # Check for ChinaCDC format indicators
        if any(k in data for k in ["NIHL1234", "NIHL346", "LEX_8h_LEX_40h_median", "creation_province"]):
            self._build_from_chinacdc(**data)
        else:
            logger.warning("无法检测数据格式，请确保传入 creation_date 字段")

    def _build_from_chinacdc(self, **data):
        """Build from ChinaCDC data format (single record)."""
        creation_date = data.get("creation_date")
        if creation_date is None:
            creation_date = datetime.now()
        elif isinstance(creation_date, str):
            creation_date = pd.to_datetime(creation_date)
        
        self.record_dates = [creation_date]
        
        # check_order 和 days_since_first 仅在显式传入时填充
        # 队列特征请通过 build_queue_features() 方法计算
        check_order_val = data.get("check_order")
        if check_order_val is not None:
            self.check_order[creation_date] = check_order_val
        
        days_val = data.get("days_since_first")
        if days_val is not None:
            self.days_since_first[creation_date] = days_val
        
        # Extract basic info
        self.staff_sex = data.get("sex")
        if self.staff_sex:
            self.staff_sex = "M" if self.staff_sex in ("Male", "男", "M", "m", "male", "1", 1) else "F"
        self.staff_age = data.get("age")
        self.staff_duration = data.get("duration")
        
        # Build StaffBasicInfo
        self.staff_basic_info[creation_date] = StaffBasicInfo(
            staff_id=self.staff_id,
            creation_date=creation_date,
            creation_province=data.get("creation_province", "未知"),
            creation_city=data.get("creation_city"),
            industry_category=data.get("industry_category"),
            employment_unit_code=data.get("employment_unit_unified_social_credit_code"),
            examination_type_code=data.get("examination_type_code"),
        )
        
        # Build StaffHealthInfo with nested auditory info
        from ohtk.staff_info.auditory_health_info import AuditoryHealthInfo
        auditory_info = None
        nihl1234 = data.get("NIHL1234")
        nihl346 = data.get("NIHL346")
        
        # 收集原始听力检测数据
        ear_data = {}
        for ear, prefix in [('left', 'L'), ('right', 'R')]:
            for freq in [500, 1000, 2000, 3000, 4000, 6000]:
                # 支持多种列名格式: L-500, L_500, left_ear_500 等
                possible_keys = [
                    f'{prefix}-{freq}',
                    f'{prefix}_{freq}',
                    f'{ear}_ear_{freq}',
                    f'{prefix}{freq}'
                ]
                for key in possible_keys:
                    if key in data and data[key] is not None:
                        ear_data[f'{prefix}-{freq}'] = data[key]
                        break
        
        # 构建 detection 数据
        detection = None
        if ear_data:
            detection = {"PTA": ear_data}
        
        if self._is_valid_number(nihl1234) or self._is_valid_number(nihl346) or detection:
            nihl_dict = {}
            if self._is_valid_number(nihl1234):
                nihl_dict["1234"] = float(nihl1234)
            if self._is_valid_number(nihl346):
                nihl_dict["346"] = float(nihl346)
            auditory_info = AuditoryHealthInfo(
                NIHL=nihl_dict if nihl_dict else None,
                norm_hearing_loss=data.get("norm_hearing_loss"),
                high_hearing_loss=data.get("high_hearing_loss"),
                detection=detection,
                ear_data=ear_data if ear_data else None,
            )
        self.staff_health_info[creation_date] = StaffHealthInfo(
            staff_id=self.staff_id,
            auditory=auditory_info,
        )
        
        # Build StaffOccHazInfo
        laeq = data.get("LEX_8h_LEX_40h_median") or data.get("LAeq")
        if laeq is not None:
            self.staff_occhaz_info[creation_date] = StaffOccHazInfo(
                staff_id=self.staff_id,
                noise_hazard_info={"LAeq": laeq}
            )

    def _build_from_dates(self, **data):
        """
        从日期索引数据构建 StaffInfo
        
        注意：check_order 和 days_since_first 不会自动计算。
        如需这些队列特征，请调用 build_queue_features() 方法。
        
        Args:
            **data: 包含列表形式数据的字典，需要 creation_date 字段
        """
        # Handle list data format
        creation_dates = data.get("creation_date", [])
        if not isinstance(creation_dates, list):
            creation_dates = [creation_dates]
        
        # Convert to datetime
        creation_dates = [pd.to_datetime(d) if isinstance(d, str) else d for d in creation_dates]
        
        if not creation_dates:
            logger.warning("没有有效的 creation_date 数据")
            return
        
        # 收集所有传入的数据字段
        all_known_fields = set(self.BASIC_INFO_FIELDS.keys()) | \
                          set(self.HEALTH_INFO_FIELDS.keys()) | \
                          set(self.OCCHAZ_INFO_FIELDS.keys()) | \
                          set(self.STAFF_DIRECT_FIELDS.keys())
        
        record_data = {}
        for key, value in data.items():
            if key in all_known_fields or key in ["check_order", "days_since_first"]:
                record_data[key] = value
            # 收集听力数据字段
            if key.startswith(('L-', 'R-', 'left_ear_', 'right_ear_')):
                record_data[key] = value
        
        # Ensure all lists have same length
        max_len = len(creation_dates)
        for k, v in record_data.items():
            if not isinstance(v, list):
                record_data[k] = [v] * max_len
        
        # Sort by date and get sorted indices
        sorted_indices = sorted(range(len(creation_dates)), key=lambda i: creation_dates[i])
        creation_dates = [creation_dates[i] for i in sorted_indices]
        for k in record_data:
            if len(record_data[k]) == max_len:
                record_data[k] = [record_data[k][i] for i in sorted_indices]
        
        self.record_dates = creation_dates
        
        # ========================================
        # check_order 和 days_since_first 仅在显式传入时填充
        # 队列特征请通过 build_queue_features() 方法计算
        # ========================================
        
        # 获取传入的 check_order 和 days_since_first（可能是 list 或 None）
        check_order_raw = record_data.get("check_order", [])
        days_since_first_raw = record_data.get("days_since_first", [])
        
        for idx, date in enumerate(creation_dates):
            # check_order: 仅使用传入值
            if check_order_raw and idx < len(check_order_raw) and check_order_raw[idx] is not None:
                self.check_order[date] = int(check_order_raw[idx])
            
            # days_since_first: 仅使用传入值
            if days_since_first_raw and idx < len(days_since_first_raw) and days_since_first_raw[idx] is not None:
                self.days_since_first[date] = int(days_since_first_raw[idx])
        
        # ========================================
        # 提取 StaffInfo 直接属性
        # ========================================
        
        # sex
        if "sex" in record_data and record_data["sex"]:
            valid_sex = [s for s in record_data["sex"] if s is not None]
            if valid_sex:
                sex_counts = pd.Series(valid_sex).value_counts()
                self.staff_sex = sex_counts.idxmax()
                if self.staff_sex:
                    self.staff_sex = "M" if self.staff_sex in ("Male", "男", "M", "m", "male", "1", 1) else "F"
        
        # name
        if "name" in record_data and record_data["name"]:
            valid_names = [n for n in record_data["name"] if n is not None and n != ""]
            if valid_names:
                name_counts = pd.Series(valid_names).value_counts()
                if not name_counts.empty:
                    self.staff_name = name_counts.idxmax()
        
        # age (use most recent)
        if "age" in record_data and record_data["age"]:
            ages = [a for a in record_data["age"] if self._is_valid_number(a)]
            if ages:
                self.staff_age = ages[-1]
        
        # duration (use most recent)
        if "duration" in record_data and record_data["duration"]:
            durations = [d for d in record_data["duration"] if self._is_valid_number(d)]
            if durations:
                self.staff_duration = durations[-1]
        
        # ========================================
        # 构建各日期的信息对象
        # ========================================
        from ohtk.staff_info.auditory_health_info import AuditoryHealthInfo
        
        for idx, date in enumerate(creation_dates):
            record = {k: v[idx] if idx < len(v) else None for k, v in record_data.items()}
            
            # StaffBasicInfo
            basic_kwargs = {"staff_id": self.staff_id, "creation_date": date}
            for src_field, dst_attr in self.BASIC_INFO_FIELDS.items():
                if src_field in record and record[src_field] is not None:
                    value = record[src_field]
                    # 处理 NaN 值
                    if isinstance(value, float) and np.isnan(value):
                        continue
                    basic_kwargs[dst_attr] = value
            if "creation_province" not in basic_kwargs:
                basic_kwargs["creation_province"] = "未知"
            
            self.staff_basic_info[date] = StaffBasicInfo(**basic_kwargs)
            
            # StaffHealthInfo
            auditory_info = None
            nihl1234 = record.get("NIHL1234")
            nihl346 = record.get("NIHL346")
            
            # 收集原始听力检测数据
            ear_data = {}
            for ear, prefix in [('left', 'L'), ('right', 'R')]:
                for freq in [500, 1000, 2000, 3000, 4000, 6000]:
                    possible_keys = [
                        f'{prefix}-{freq}',
                        f'{prefix}_{freq}',
                        f'{ear}_ear_{freq}',
                        f'{prefix}{freq}'
                    ]
                    for key in possible_keys:
                        if key in record and record[key] is not None:
                            value = record[key]
                            # 处理 NaN
                            if isinstance(value, float) and np.isnan(value):
                                continue
                            ear_data[f'{prefix}-{freq}'] = value
                            break
            
            # 构建 detection 数据
            detection = None
            if ear_data:
                detection = {"PTA": ear_data}
            
            if self._is_valid_number(nihl1234) or self._is_valid_number(nihl346) or detection:
                nihl_dict = {}
                if self._is_valid_number(nihl1234):
                    nihl_dict["1234"] = float(nihl1234)
                if self._is_valid_number(nihl346):
                    nihl_dict["346"] = float(nihl346)
                auditory_info = AuditoryHealthInfo(
                    detection=detection,
                    diagnose=record.get("auditory_diagnose"),
                    NIHL=nihl_dict if nihl_dict else None,
                    ear_data=ear_data if ear_data else None,
                )
            self.staff_health_info[date] = StaffHealthInfo(
                staff_id=self.staff_id,
                auditory=auditory_info,
            )
            
            # StaffOccHazInfo
            noise_info = record.get("noise_hazard_info")
            laeq = record.get("LEX_8h_LEX_40h_median") or record.get("LAeq")
            if noise_info or laeq:
                if laeq and not noise_info:
                    noise_info = {"LAeq": laeq}
                self.staff_occhaz_info[date] = StaffOccHazInfo(
                    staff_id=self.staff_id,
                    noise_hazard_info=noise_info
                )

    @staticmethod
    def _is_valid_number(val) -> bool:
        """检查值是否为有效数字（非 None、非 NaN）"""
        if val is None:
            return False
        try:
            return not np.isnan(val)
        except (TypeError, ValueError):
            return val is not None

    # ========================================
    # 类方法：帮助和字段信息
    # ========================================

    @classmethod
    def get_expected_fields(cls) -> Dict[str, List[str]]:
        """
        获取 StaffInfo 支持的所有字段
        
        Returns:
            字典，按类别分组的字段列表
        """
        return {
            "必需字段": ["staff_id", "creation_date"],
            "员工直接属性": list(cls.STAFF_DIRECT_FIELDS.keys()),
            "基础信息字段 (StaffBasicInfo)": list(cls.BASIC_INFO_FIELDS.keys()),
            "健康信息字段 (StaffHealthInfo)": list(cls.HEALTH_INFO_FIELDS.keys()),
            "职业危害字段 (StaffOccHazInfo)": list(cls.OCCHAZ_INFO_FIELDS.keys()),
            "时间序列字段 (可选，自动计算)": ["check_order", "days_since_first"],
        }

    @classmethod
    def help(cls) -> str:
        """
        打印 StaffInfo 的帮助信息
        
        Returns:
            帮助信息字符串
        """
        help_text = """
================================================================================
StaffInfo 使用帮助
================================================================================

【概述】
StaffInfo 是职业健康检查员工信息的统一模型，支持时间序列数据。

【数据加载方式】

1. 直接传入字典（单条记录）:
   staff = StaffInfo(
       staff_id="W001",
       creation_date="2024-01-15",
       sex=1, age=35, duration=10,
       NIHL346=18.5, LAeq=85.0
   )

2. 直接传入字典（多条记录，列表形式）:
   staff = StaffInfo(
       staff_id="W001",
       creation_date=["2024-01-15", "2025-01-20"],
       sex=[1, 1], age=[35, 36],
       NIHL346=[18.5, 20.0], LAeq=[85.0, 86.0]
   )

3. 从 DataFrame 加载单个工人:
   staff = StaffInfo.load_from_dataframe(df, worker_id="W001")

4. 批量加载多个工人:
   staff_dict = StaffInfo.load_batch_from_dataframe(df)

【支持的字段】
"""
        fields = cls.get_expected_fields()
        for category, field_list in fields.items():
            help_text += f"\n  {category}:\n"
            for field in field_list:
                help_text += f"    - {field}\n"
        
        help_text += """
【队列数据聚合】
check_order 和 days_since_first 不会在数据加载时自动计算。
如需这些队列特征，请调用 build_queue_features() 方法：

    # 单个工人
    staff.build_queue_features()
    
    # 批量处理
    StaffInfo.build_queue_features_batch(staff_dict)

【完整工作流示例】

    # 步骤 1: 加载数据
    staff_dict = StaffInfo.load_batch_from_dataframe(df)
    
    # 步骤 2: 计算队列特征（check_order, days_since_first）
    StaffInfo.build_queue_features_batch(staff_dict)
    
    # 步骤 3: 转换为分析用 DataFrame
    analysis_df = StaffInfo.to_analysis_dataframe_batch(staff_dict)

【字段映射】
外部数据字段会自动映射到内部属性，例如：
- LEX_8h_LEX_40h_median -> LAeq
- employment_unit_unified_social_credit_code -> employment_unit_code
- cigaretee_per_day -> cigarette_per_day

查看完整映射: StaffInfo.BASIC_INFO_FIELDS, StaffInfo.HEALTH_INFO_FIELDS 等

================================================================================
"""
        print(help_text)
        return help_text

    # ========================================
    # 数据加载类方法
    # ========================================

    @classmethod
    def load_from_dataframe(
        cls, 
        df: pd.DataFrame, 
        worker_id: str,
        field_mapping: Optional[Dict[str, str]] = None
    ) -> "StaffInfo":
        """
        从 DataFrame 加载单个工人的 StaffInfo
        
        Args:
            df: DataFrame，包含工人记录
            worker_id: 工人ID
            field_mapping: 自定义字段映射 {df列名: StaffInfo字段名}
            
        Returns:
            StaffInfo 实例
            
        Example:
            # 使用默认映射
            staff = StaffInfo.load_from_dataframe(df, "W001")
            
            # 使用自定义映射
            staff = StaffInfo.load_from_dataframe(
                df, "W001",
                field_mapping={"my_date_col": "creation_date", "noise_level": "LAeq"}
            )
        """
        # 支持 staff_id 或 worker_id 列
        id_column = None
        if "staff_id" in df.columns:
            id_column = "staff_id"
        elif "worker_id" in df.columns:
            id_column = "worker_id"
        else:
            raise ValueError("DataFrame 必须包含 'staff_id' 或 'worker_id' 列")
        
        worker_df = df[df[id_column] == worker_id].copy()
        if worker_df.empty:
            raise ValueError(f"未找到 {id_column}: {worker_id} 的记录")
        
        # 应用自定义字段映射
        if field_mapping:
            worker_df = worker_df.rename(columns=field_mapping)
        
        # Sort by date
        if "creation_date" in worker_df.columns:
            worker_df["creation_date"] = pd.to_datetime(worker_df["creation_date"])
            worker_df = worker_df.sort_values("creation_date")
        
        # Build data dict
        data = {"staff_id": worker_id}
        
        # 收集所有可能的字段
        all_fields = (
            ["creation_date", "check_order", "days_since_first"] +
            list(cls.STAFF_DIRECT_FIELDS.keys()) +
            list(cls.BASIC_INFO_FIELDS.keys()) +
            list(cls.HEALTH_INFO_FIELDS.keys()) +
            list(cls.OCCHAZ_INFO_FIELDS.keys())
        )
        
        for col in all_fields:
            if col in worker_df.columns:
                data[col] = worker_df[col].tolist()
        
        # 收集听力数据字段 (L-500, R-500, etc.)
        for col in worker_df.columns:
            if col.startswith(('L-', 'R-', 'left_ear_', 'right_ear_')):
                data[col] = worker_df[col].tolist()
        
        return cls(**data)

    @classmethod
    def load_batch_from_dataframe(
        cls, 
        df: pd.DataFrame,
        field_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, "StaffInfo"]:
        """
        从 DataFrame 批量加载多个工人的 StaffInfo
        
        Args:
            df: DataFrame，包含多个工人的记录
            field_mapping: 自定义字段映射 {df列名: StaffInfo字段名}
            
        Returns:
            {worker_id: StaffInfo} 字典
        """
        if "worker_id" not in df.columns:
            raise ValueError("DataFrame 必须包含 'worker_id' 列")
        
        result = {}
        for worker_id in df["worker_id"].unique():
            try:
                result[worker_id] = cls.load_from_dataframe(df, worker_id, field_mapping)
            except Exception as e:
                logger.warning(f"加载工人 {worker_id} 失败: {e}")
        
        return result

    # ========================================
    # 实例方法
    # ========================================

    def get_most_recent_date(self) -> Optional[datetime]:
        """获取最近一次检查日期"""
        if self.record_dates:
            return max(self.record_dates)
        return None

    def get_record_key(self, date: Optional[datetime] = None) -> Optional[datetime]:
        """获取记录键（日期）"""
        if date:
            if date in self.staff_basic_info:
                return date
            # Try to find matching year
            for d in self.record_dates:
                if d.year == date.year:
                    return d
        
        # Return most recent
        if self.record_dates:
            return max(self.record_dates)
        return None

    # ========================================
    # 队列数据聚合方法
    # ========================================

    def build_queue_features(self, force_recalculate: bool = False) -> None:
        """
        计算队列研究所需的时间特征
        
        计算内容:
        - check_order: 按 creation_date 排序的检查序号（从 1 开始）
        - days_since_first: 距离首次检查的天数
        
        Args:
            force_recalculate: 是否强制重新计算（即使已有值）
        
        示例:
            staff = StaffInfo(staff_id="W001", creation_date=[...], ...)
            staff.build_queue_features()  # 计算队列特征
        """
        if not self.record_dates:
            logger.warning(f"工人 {self.staff_id} 没有检查记录，跳过队列特征计算")
            return
        
        # 确保日期已排序
        sorted_dates = sorted(self.record_dates)
        first_date = sorted_dates[0]
        
        for idx, date in enumerate(sorted_dates):
            # check_order
            if force_recalculate or date not in self.check_order:
                self.check_order[date] = idx + 1
            
            # days_since_first
            if force_recalculate or date not in self.days_since_first:
                self.days_since_first[date] = (date - first_date).days
        
        logger.debug(f"工人 {self.staff_id} 队列特征计算完成：{len(sorted_dates)} 条记录")

    @classmethod
    def build_queue_features_batch(
        cls,
        staff_dict: Dict[str, "StaffInfo"],
        force_recalculate: bool = False
    ) -> Dict[str, "StaffInfo"]:
        """
        批量计算队列特征
        
        Args:
            staff_dict: worker_id -> StaffInfo 映射
            force_recalculate: 是否强制重新计算
        
        Returns:
            更新后的 staff_dict（原地修改）
        
        示例:
            staff_dict = StaffInfo.load_batch_from_dataframe(df)
            StaffInfo.build_queue_features_batch(staff_dict)
        """
        count = 0
        for worker_id, staff in staff_dict.items():
            staff.build_queue_features(force_recalculate)
            count += 1
        
        logger.info(f"批量队列特征计算完成：处理了 {count} 个工人")
        return staff_dict

    # ========================================
    # DataFrame 转换方法
    # ========================================

    def to_analysis_dataframe(self, include_fields: List[str] = None) -> pd.DataFrame:
        """
        将 StaffInfo 转换为分析用 DataFrame（单个工人）
        
        Args:
            include_fields: 要包含的额外字段列表（如 ["creation_province", "creation_city"]）
        
        Returns:
            DataFrame，每行一次检查记录
        
        包含字段:
        - worker_id (staff_id)
        - creation_date
        - check_order (如果已计算)
        - days_since_first (如果已计算)
        - sex, age, duration (StaffInfo 属性)
        - NIHL1234, NIHL346 (健康信息)
        - LAeq (职业危害信息)
        
        示例:
            staff.build_queue_features()  # 先计算队列特征
            df = staff.to_analysis_dataframe()
        """
        if not self.record_dates:
            logger.warning(f"工人 {self.staff_id} 没有检查记录")
            return pd.DataFrame()
        
        # 检查队列特征是否已计算
        if not self.check_order or not self.days_since_first:
            logger.warning(
                f"工人 {self.staff_id} 的队列特征未计算，建议先调用 build_queue_features() 方法"
            )
        
        records = []
        for date in sorted(self.record_dates):
            record = {
                "worker_id": self.staff_id,
                "creation_date": date,
                "sex": 1 if self.staff_sex == "M" else (0 if self.staff_sex == "F" else None),
                "age": self.staff_age,
                "duration": self.staff_duration,
            }
            
            # 添加队列特征（如果已计算）
            if self.check_order:
                record["check_order"] = self.check_order.get(date)
            if self.days_since_first:
                record["days_since_first"] = self.days_since_first.get(date)
            
            # 从健康信息获取 NIHL
            health_info = self.staff_health_info.get(date)
            if health_info and health_info.auditory:
                if health_info.auditory.NIHL:
                    record["NIHL1234"] = health_info.auditory.NIHL.get("1234")
                    record["NIHL346"] = health_info.auditory.NIHL.get("346")
                record["norm_hearing_loss"] = health_info.auditory.norm_hearing_loss
                record["high_hearing_loss"] = health_info.auditory.high_hearing_loss
            
            # 从职业危害信息获取 LAeq
            occhaz_info = self.staff_occhaz_info.get(date)
            if occhaz_info and occhaz_info.noise_hazard_info:
                if isinstance(occhaz_info.noise_hazard_info, dict):
                    record["LAeq"] = occhaz_info.noise_hazard_info.get("LAeq")
                elif hasattr(occhaz_info.noise_hazard_info, "LAeq"):
                    record["LAeq"] = occhaz_info.noise_hazard_info.LAeq
            
            # 从基础信息获取额外字段
            if include_fields:
                basic_info = self.staff_basic_info.get(date)
                if basic_info:
                    for field in include_fields:
                        if hasattr(basic_info, field):
                            record[field] = getattr(basic_info, field)
            
            records.append(record)
        
        return pd.DataFrame(records)

    @classmethod
    def to_analysis_dataframe_batch(
        cls,
        staff_dict: Dict[str, "StaffInfo"],
        include_fields: List[str] = None
    ) -> pd.DataFrame:
        """
        批量将多个 StaffInfo 转换为分析用 DataFrame
        
        Args:
            staff_dict: worker_id -> StaffInfo 映射
            include_fields: 要包含的额外字段列表
        
        Returns:
            合并后的 DataFrame（所有工人）
        
        示例:
            staff_dict = StaffInfo.load_batch_from_dataframe(df)
            StaffInfo.build_queue_features_batch(staff_dict)
            analysis_df = StaffInfo.to_analysis_dataframe_batch(staff_dict)
        """
        dfs = []
        for worker_id, staff in staff_dict.items():
            df = staff.to_analysis_dataframe(include_fields)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            logger.warning("没有有效的数据可转换")
            return pd.DataFrame()
        
        result = pd.concat(dfs, ignore_index=True)
        logger.info(f"批量转换完成：{len(staff_dict)} 个工人，{len(result)} 条记录")
        return result

    def _get_staff_data_for_nipts(self, **kwargs) -> tuple:
        """获取 NIPTS 预测所需的员工数据"""
        recent_key = self.get_record_key()
        
        # LAeq
        LAeq = kwargs.pop("LAeq", None)
        if LAeq is None and recent_key is not None:
            occhaz_info = self.staff_occhaz_info.get(recent_key)
            if occhaz_info and occhaz_info.noise_hazard_info:
                if hasattr(occhaz_info.noise_hazard_info, 'LAeq'):
                    LAeq = occhaz_info.noise_hazard_info.LAeq
                elif isinstance(occhaz_info.noise_hazard_info, dict):
                    LAeq = occhaz_info.noise_hazard_info.get('LAeq')
        
        # age
        age = kwargs.pop("age", None)
        if age is None:
            age = self.staff_age
        
        # sex
        sex = kwargs.pop("sex", None)
        if sex is None:
            sex = self.staff_sex
        
        # duration
        duration = kwargs.pop("duration", None)
        if duration is None:
            duration = self.staff_duration
        
        return LAeq, age, sex, duration, kwargs

    def NIPTS_predict_iso1999_2013(self,
                                   Hs: bool = False,
                                   percentrage: int = 50,
                                   mean_key: list = None,
                                   standard: str = "Chinese",
                                   **kwargs) -> float:
        """根据ISO 1999:2013标准预测噪声性永久阈移(NIPTS)"""
        from ohtk.diagnose_info import AuditoryDiagnose
        
        if mean_key is None:
            mean_key = [3000, 4000, 6000]
        
        LAeq, age, sex, duration, remaining_kwargs = self._get_staff_data_for_nipts(**kwargs)
        NH_limit = remaining_kwargs.pop("NH_limit", True)

        if LAeq is None or age is None or sex is None or duration is None:
            raise ValueError("NIPTS 预测需要 LAeq, age, sex, duration 数据")

        return AuditoryDiagnose.predict_NIPTS_iso1999_2013(
            LAeq=LAeq, age=age, sex=sex, duration=duration,
            Hs=Hs, percentrage=percentrage, mean_key=mean_key,
            standard=standard, NH_limit=NH_limit
        )

    def NIPTS_predict_iso1999_2023(self,
                                   percentrage: int = 50,
                                   mean_key: list = None,
                                   **kwargs) -> float:
        """根据ISO 1999:2023标准预测噪声性永久阈移(NIPTS)"""
        from ohtk.diagnose_info import AuditoryDiagnose
        
        if mean_key is None:
            mean_key = [3000, 4000, 6000]
        
        LAeq, age, sex, duration, remaining_kwargs = self._get_staff_data_for_nipts(**kwargs)
        extrapolation = remaining_kwargs.pop("extrapolation", None)
        NH_limit = remaining_kwargs.pop("NH_limit", True)

        if LAeq is None or age is None or sex is None or duration is None:
            raise ValueError("NIPTS 预测需要 LAeq, age, sex, duration 数据")

        return AuditoryDiagnose.predict_NIPTS_iso1999_2023(
            LAeq=LAeq, age=age, sex=sex, duration=duration,
            percentrage=percentrage, mean_key=mean_key,
            extrapolation=extrapolation, NH_limit=NH_limit
        )

    def calculate_auditory_nihl(
        self,
        date: Optional[datetime] = None,
        freq_keys: List[str] = None,
        apply_correction: bool = False
    ) -> Dict[str, float]:
        """计算指定日期的听力 NIHL 值"""
        if freq_keys is None:
            freq_keys = ["1234", "346"]
            
        record_key = self.get_record_key(date)
        if record_key is None:
            raise ValueError("未找到健康记录")
        
        health_info = self.staff_health_info.get(record_key)
        if health_info is None or health_info.auditory is None:
            raise ValueError(f"未找到 {record_key} 的听力健康信息")
        
        return health_info.auditory.calculate_NIHL(
            sex=self.staff_sex,
            age=self.staff_age,
            freq_keys=freq_keys,
            apply_correction=apply_correction
        )
