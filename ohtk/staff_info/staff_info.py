import pandas as pd
import numpy as np
from pydantic import BaseModel
from functional import seq
from datetime import datetime
from typing import Dict, Union, Optional, List
from loguru import logger

from ohtk.staff_info.staff_basic_info import StaffBasicInfo
from ohtk.staff_info.staff_health_info import StaffHealthInfo
from ohtk.staff_info.staff_occhaz_info import StaffOccHazInfo


class StaffInfo(BaseModel):
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
    
    # Legacy support
    record_years: list = []  # Kept for backward compatibility
    
    conflict_attr: list = []
    current_year: int = datetime.now().year

    def __init__(self, **data):
        # Extract check_order and days_since_first before Pydantic validation
        # to avoid type errors (they may be int or list, not dict)
        _check_order_raw = data.pop("check_order", None)
        _days_since_first_raw = data.pop("days_since_first", None)
        
        super().__init__(**data)
        
        # Store raw values for use in build methods
        data["check_order"] = _check_order_raw
        data["days_since_first"] = _days_since_first_raw
        
        # Detect data format and build accordingly
        if "creation_date" in data or self._has_datetime_data(data):
            self._build_from_dates(**data)
        elif "record_year" in data:
            self._build_from_years(**data)
        else:
            # Try to auto-detect format
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
            # Default to legacy format if no specific indicators
            logger.warning("Could not detect data format, attempting legacy build")
            self._build_from_years(**data)

    def _build_from_chinacdc(self, **data):
        """Build from ChinaCDC data format (single record)."""
        # Handle single record or list of records
        creation_date = data.get("creation_date")
        if creation_date is None:
            creation_date = datetime.now()
        elif isinstance(creation_date, str):
            creation_date = pd.to_datetime(creation_date)
        
        self.record_dates = [creation_date]
        self.check_order[creation_date] = data.get("check_order", 1)
        self.days_since_first[creation_date] = data.get("days_since_first", 0)
        
        # Extract basic info
        self.staff_sex = data.get("sex")
        if self.staff_sex:
            self.staff_sex = "M" if self.staff_sex in ("Male", "男", "M", "m", "male", "1", 1) else "F"
        self.staff_age = data.get("age")
        self.staff_duration = data.get("duration")
        
        # Build StaffBasicInfo (sex/age/duration stored only in StaffInfo)
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
        # Handle pandas NaN values
        def is_valid_number(val):
            if val is None:
                return False
            try:
                return not np.isnan(val)
            except (TypeError, ValueError):
                return val is not None
        
        if is_valid_number(nihl1234) or is_valid_number(nihl346):
            # Build NIHL dict, filtering out None/NaN values
            nihl_dict = {}
            if is_valid_number(nihl1234):
                nihl_dict["1234"] = float(nihl1234)
            if is_valid_number(nihl346):
                nihl_dict["346"] = float(nihl346)
            auditory_info = AuditoryHealthInfo(
                NIHL=nihl_dict if nihl_dict else None,
                norm_hearing_loss=data.get("norm_hearing_loss"),
                high_hearing_loss=data.get("high_hearing_loss"),
            )
        self.staff_health_info[creation_date] = StaffHealthInfo(
            staff_id=self.staff_id,
            auditory=auditory_info,
        )
        
        # Build StaffOccHazInfo - use LEX_8h_LEX_40h_median as LAeq
        laeq = data.get("LEX_8h_LEX_40h_median") or data.get("LAeq")
        if laeq is not None:
            self.staff_occhaz_info[creation_date] = StaffOccHazInfo(
                staff_id=self.staff_id,
                noise_hazard_info={"LAeq": laeq}
            )

    def _build_from_dates(self, **data):
        """Build from data with creation_date as index."""
        # Handle list data format
        creation_dates = data.get("creation_date", [])
        if not isinstance(creation_dates, list):
            creation_dates = [creation_dates]
        
        # Convert to datetime
        creation_dates = [pd.to_datetime(d) if isinstance(d, str) else d for d in creation_dates]
        
        # Build record list
        record_fields = [
            "name", "factory_name", "work_shop", "work_position", "sex", "age",
            "duration", "smoking", "year_of_smoking", "cigaretee_per_day",
            "occupational_clinic_class", "auditory_detection", "auditory_diagnose",
            "noise_hazard_info", "creation_province", "creation_city", 
            "industry_category", "employment_unit_unified_social_credit_code",
            "examination_type_code", "NIHL1234", "NIHL346", "LEX_8h_LEX_40h_median"
        ]
        
        record_data = {k: data.get(k, []) for k in record_fields if k in data}
        
        # Ensure all lists have same length
        max_len = len(creation_dates)
        for k, v in record_data.items():
            if not isinstance(v, list):
                record_data[k] = [v] * max_len
        
        # Sort by date
        sorted_indices = sorted(range(len(creation_dates)), key=lambda i: creation_dates[i])
        creation_dates = [creation_dates[i] for i in sorted_indices]
        for k in record_data:
            if len(record_data[k]) == max_len:
                record_data[k] = [record_data[k][i] for i in sorted_indices]
        
        self.record_dates = creation_dates
        
        # Calculate time-series features
        first_date = creation_dates[0] if creation_dates else datetime.now()
        for idx, date in enumerate(creation_dates):
            self.check_order[date] = idx + 1
            self.days_since_first[date] = (date - first_date).days
        
        # Extract common info
        if "sex" in record_data and record_data["sex"]:
            sex_counts = pd.Series(record_data["sex"]).value_counts()
            self.staff_sex = sex_counts.idxmax() if len(sex_counts) > 0 else None
            if self.staff_sex:
                self.staff_sex = "M" if self.staff_sex in ("Male", "男", "M", "m", "male", "1", 1) else "F"
        
        if "name" in record_data and record_data["name"]:
            name_counts = pd.Series(record_data["name"]).value_counts()
            self.staff_name = name_counts.idxmax() if len(name_counts) > 0 else ""
        
        # Extract age (use most recent value)
        if "age" in record_data and record_data["age"]:
            ages = [a for a in record_data["age"] if a is not None]
            if ages:
                self.staff_age = ages[-1]  # Use most recent age
        
        # Extract duration (use most recent value)
        if "duration" in record_data and record_data["duration"]:
            durations = [d for d in record_data["duration"] if d is not None]
            if durations:
                self.staff_duration = durations[-1]  # Use most recent duration
        
        # Build info objects for each date
        from ohtk.staff_info.auditory_health_info import AuditoryHealthInfo
        for idx, date in enumerate(creation_dates):
            record = {k: v[idx] if idx < len(v) else None for k, v in record_data.items()}
            
            # StaffBasicInfo (sex/age/duration stored only in StaffInfo)
            self.staff_basic_info[date] = StaffBasicInfo(
                staff_id=self.staff_id,
                name=self.staff_name,
                factory_name=record.get("factory_name"),
                work_shop=record.get("work_shop"),
                work_position=record.get("work_position"),
                smoking=record.get("smoking"),
                year_of_smoking=record.get("year_of_smoking"),
                cigarette_per_day=record.get("cigaretee_per_day"),
                occupational_clinic_class=record.get("occupational_clinic_class"),
                creation_date=date,
                creation_province=record.get("creation_province", "未知"),
                creation_city=record.get("creation_city"),
                industry_category=record.get("industry_category"),
                employment_unit_code=record.get("employment_unit_unified_social_credit_code"),
                examination_type_code=record.get("examination_type_code"),
            )
            
            # StaffHealthInfo with nested auditory info
            auditory_info = None
            nihl1234 = record.get("NIHL1234")
            nihl346 = record.get("NIHL346")
            # Handle pandas NaN values (np.isnan raises TypeError for non-numeric types)
            def is_valid_number(val):
                if val is None:
                    return False
                try:
                    return not np.isnan(val)
                except (TypeError, ValueError):
                    return val is not None
            
            if is_valid_number(nihl1234) or is_valid_number(nihl346) or record.get("auditory_detection"):
                # Build NIHL dict, filtering out None/NaN values
                nihl_dict = {}
                if is_valid_number(nihl1234):
                    nihl_dict["1234"] = float(nihl1234)
                if is_valid_number(nihl346):
                    nihl_dict["346"] = float(nihl346)
                auditory_info = AuditoryHealthInfo(
                    detection=record.get("auditory_detection"),
                    NIHL=nihl_dict if nihl_dict else None,
                )
            self.staff_health_info[date] = StaffHealthInfo(
                staff_id=self.staff_id,
                auditory=auditory_info,
            )
            
            noise_info = record.get("noise_hazard_info")
            laeq = record.get("LEX_8h_LEX_40h_median")
            if noise_info or laeq:
                if laeq and not noise_info:
                    noise_info = {"LAeq": laeq}
                self.staff_occhaz_info[date] = StaffOccHazInfo(
                    staff_id=self.staff_id,
                    noise_hazard_info=noise_info
                )

    def _build_from_years(self, **data):
        """从年份格式数据构建（旧格式兼容）"""
        from ohtk.staff_info.auditory_health_info import AuditoryHealthInfo
        
        # data中的信息应当是json形式，体检记录相关的字段下的类型为list
        record_mesg_df = seq(data.items()).filter(lambda x: x[0] in [
            "record_year", "name", "factory_name", "work_shop", "work_position", "sex", "age",
            "duration", "smoking", "year_of_smoking", "cigaretee_per_day",
            "occupational_clinic_class", "auditory_detection", "auditory_diagnose",
            "noise_hazard_info"]).dict()
        try:
            record_mesg_df = pd.DataFrame(
                record_mesg_df).set_index("record_year")
        except ValueError:
            logger.error("列表不等长，体检记录的内容存在缺失，请检查！")
            raise
        # 常量信息计算与校验
        staff_name = record_mesg_df["name"].value_counts()
        staff_sex = record_mesg_df["sex"].value_counts()
        staff_age = (
            record_mesg_df["age"] - record_mesg_df.index + self.current_year).value_counts()
        staff_duration = (
            record_mesg_df["duration"] - record_mesg_df.index + self.current_year).value_counts()
        for attr_name, attr in zip(["staff_name", "staff_sex", "staff_age", "staff_duration"],
                                   [staff_name, staff_sex, staff_age, staff_duration]):
            if len(attr) > 1:
                logger.warning(f"注意！根据记录得到的{attr_name}不唯一！将使用频数最高的结果。")
                self.conflict_attr.append(attr_name)

        self.staff_name = staff_name.idxmax()
        self.staff_sex = staff_sex.idxmax()
        self.staff_age = staff_age.idxmax()
        self.staff_duration = staff_duration.idxmax()

        # 体检记录信息逐条写入
        record_mesgs = record_mesg_df.to_dict(orient="index")
        staff_basic_info = {}
        staff_health_info = {}
        staff_occhaz_info = {}
        for record_year, record_mesg in record_mesgs.items():
            self.record_years.append(record_year)
            # 构建日期（用于新结构兼容）
            record_date = datetime(record_year, 1, 1)
            
            # 员工基础信息构建（不再传递 sex/age/duration）
            staff_basic_info[record_year] = StaffBasicInfo(
                staff_id=self.staff_id,
                name=self.staff_name,
                factory_name=record_mesg.get("factory_name"),
                work_shop=record_mesg.get("work_shop"),
                work_position=record_mesg.get("work_position"),
                smoking=record_mesg.get("smoking"),
                year_of_smoking=record_mesg.get("year_of_smoking"),
                cigarette_per_day=record_mesg.get("cigaretee_per_day"),
                occupational_clinic_class=record_mesg.get("occupational_clinic_class"),
                creation_date=record_date,
                creation_province="未知",  # 旧格式无此字段
            )
            
            # 员工健康诊断信息构建（使用嵌套结构）
            auditory_info = None
            if record_mesg.get("auditory_detection") or record_mesg.get("auditory_diagnose"):
                auditory_info = AuditoryHealthInfo(
                    detection=record_mesg.get("auditory_detection"),
                    diagnose=record_mesg.get("auditory_diagnose"),
                )
            staff_health_info[record_year] = StaffHealthInfo(
                staff_id=self.staff_id,
                auditory=auditory_info,
            )
            
            # 员工职业危害因素信息构建
            staff_occhaz_info[record_year] = StaffOccHazInfo(
                staff_id=self.staff_id,
                noise_hazard_info=record_mesg.get("noise_hazard_info")
            )

        self.staff_basic_info = staff_basic_info
        self.staff_health_info = staff_health_info
        self.staff_occhaz_info = staff_occhaz_info
        
        # Also populate record_dates for consistency
        for record_year in self.record_years:
            date = datetime(record_year, 1, 1)
            self.record_dates.append(date)
            self.check_order[date] = self.record_years.index(record_year) + 1
            first_year = min(self.record_years)
            self.days_since_first[date] = (record_year - first_year) * 365

    @classmethod
    def load_from_dataframe(cls, df: pd.DataFrame, worker_id: str) -> "StaffInfo":
        """
        Load StaffInfo from a DataFrame for a specific worker.
        
        Supports ChinaCDC data format with columns like:
        - worker_id, creation_date, sex, age
        - NIHL1234, NIHL346, LEX_8h_LEX_40h_median
        - creation_province, creation_city, industry_category
        
        Args:
            df: DataFrame containing worker records
            worker_id: The worker ID to filter
            
        Returns:
            StaffInfo instance with time-series data
        """
        worker_df = df[df["worker_id"] == worker_id].copy()
        if worker_df.empty:
            raise ValueError(f"No records found for worker_id: {worker_id}")
        
        # Sort by date
        if "creation_date" in worker_df.columns:
            worker_df["creation_date"] = pd.to_datetime(worker_df["creation_date"])
            worker_df = worker_df.sort_values("creation_date")
        
        # Build data dict
        data = {"staff_id": worker_id}
        
        # List columns
        list_columns = [
            "creation_date", "sex", "age", "duration",
            "NIHL1234", "NIHL346", "LEX_8h_LEX_40h_median",
            "creation_province", "creation_city", "industry_category",
            "employment_unit_unified_social_credit_code", "examination_type_code",
            "check_order", "days_since_first"
        ]
        
        for col in list_columns:
            if col in worker_df.columns:
                data[col] = worker_df[col].tolist()
        
        return cls(**data)

    @classmethod
    def load_batch_from_dataframe(cls, df: pd.DataFrame) -> Dict[str, "StaffInfo"]:
        """
        Load multiple StaffInfo instances from a DataFrame.
        
        Args:
            df: DataFrame containing records for multiple workers
            
        Returns:
            Dictionary mapping worker_id to StaffInfo instance
        """
        if "worker_id" not in df.columns:
            raise ValueError("DataFrame must contain 'worker_id' column")
        
        result = {}
        for worker_id in df["worker_id"].unique():
            try:
                result[worker_id] = cls.load_from_dataframe(df, worker_id)
            except Exception as e:
                logger.warning(f"Failed to load worker {worker_id}: {e}")
        
        return result

    def get_most_recent_date(self) -> Optional[datetime]:
        """Get the most recent record date."""
        if self.record_dates:
            return max(self.record_dates)
        elif self.record_years:
            return datetime(max(self.record_years), 1, 1)
        return None

    def get_record_key(self, date: Optional[datetime] = None) -> Union[datetime, int, None]:
        """Get the appropriate record key (datetime or year)."""
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
        elif self.record_years:
            return max(self.record_years)
        return None

    def _get_staff_data_for_nipts(self, **kwargs) -> tuple:
        """Helper method to get staff data for NIPTS prediction.
        
        Returns:
            tuple: (LAeq, age, sex, duration, remaining_kwargs)
        """
        # Get the most recent record key
        recent_key = self.get_record_key()
        
        # Try to get LAeq
        LAeq = kwargs.pop("LAeq", None)
        if LAeq is None and recent_key is not None:
            occhaz_info = self.staff_occhaz_info.get(recent_key)
            if occhaz_info and occhaz_info.noise_hazard_info:
                if hasattr(occhaz_info.noise_hazard_info, 'LAeq'):
                    LAeq = occhaz_info.noise_hazard_info.LAeq
                elif isinstance(occhaz_info.noise_hazard_info, dict):
                    LAeq = occhaz_info.noise_hazard_info.get('LAeq')
        
        # Try to get age
        age = kwargs.pop("age", None)
        if age is None and recent_key is not None:
            basic_info = self.staff_basic_info.get(recent_key)
            if basic_info:
                age = basic_info.age
        if age is None:
            age = self.staff_age
        
        # Try to get sex
        sex = kwargs.pop("sex", None)
        if sex is None and recent_key is not None:
            basic_info = self.staff_basic_info.get(recent_key)
            if basic_info:
                sex = basic_info.sex
        if sex is None:
            sex = self.staff_sex
        
        # Try to get duration
        duration = kwargs.pop("duration", None)
        if duration is None and recent_key is not None:
            basic_info = self.staff_basic_info.get(recent_key)
            if basic_info:
                duration = basic_info.duration
        if duration is None:
            duration = self.staff_duration
        
        return LAeq, age, sex, duration, kwargs

    def NIPTS_predict_iso1999_2013(self,
                                   Hs: bool = False,
                                   percentrage: int = 50,
                                   mean_key: list = None,
                                   standard: str = "Chinese",
                                   **kwargs) -> float:
        """根据ISO 1999:2013标准预测噪声性永久阈移(NIPTS)
        
        便捷接口，委托给 AuditoryDiagnose.predict_NIPTS_iso1999_2013()
        
        Args:
            Hs: 是否考虑基底听力损失
            percentrage: 百分位数，默认50
            mean_key: 频率列表，默认[3000, 4000, 6000]
            standard: 标准类型，"Chinese"或其他
            **kwargs: 可选参数，如LAeq, age, sex, duration, NH_limit
            
        Returns:
            float: 预测的NIPTS值
            
        Raises:
            ValueError: 当缺少必要数据时
        """
        from ohtk.diagnose_info import AuditoryDiagnose
        
        if mean_key is None:
            mean_key = [3000, 4000, 6000]
        
        # Get staff data using helper method
        LAeq, age, sex, duration, remaining_kwargs = self._get_staff_data_for_nipts(**kwargs)
        NH_limit = remaining_kwargs.pop("NH_limit", True)

        # Check if required data is available
        if LAeq is None or age is None or sex is None or duration is None:
            raise ValueError("Required data (LAeq, age, sex, or duration) is missing for NIPTS prediction")

        return AuditoryDiagnose.predict_NIPTS_iso1999_2013(
            LAeq=LAeq,
            age=age,
            sex=sex,
            duration=duration,
            Hs=Hs,
            percentrage=percentrage,
            mean_key=mean_key,
            standard=standard,
            NH_limit=NH_limit
        )

    def NIPTS_predict_iso1999_2023(self,
                                   percentrage: int = 50,
                                   mean_key: list = None,
                                   **kwargs) -> float:
        """根据ISO 1999:2023标准预测噪声性永久阈移(NIPTS)
        
        便捷接口，委托给 AuditoryDiagnose.predict_NIPTS_iso1999_2023()
        
        Args:
            percentrage: 百分位数，默认50
            mean_key: 频率列表，默认[3000, 4000, 6000]
            **kwargs: 可选参数，如LAeq, age, sex, duration, extrapolation, NH_limit
            
        Returns:
            float: 预测的NIPTS值
            
        Raises:
            ValueError: 当缺少必要数据时
        """
        from ohtk.diagnose_info import AuditoryDiagnose
        
        if mean_key is None:
            mean_key = [3000, 4000, 6000]
        
        # Get staff data using helper method
        LAeq, age, sex, duration, remaining_kwargs = self._get_staff_data_for_nipts(**kwargs)
        extrapolation = remaining_kwargs.pop("extrapolation", None)
        NH_limit = remaining_kwargs.pop("NH_limit", True)

        # Check if required data is available
        if LAeq is None or age is None or sex is None or duration is None:
            raise ValueError("Required data (LAeq, age, sex, or duration) is missing for NIPTS prediction")

        return AuditoryDiagnose.predict_NIPTS_iso1999_2023(
            LAeq=LAeq,
            age=age,
            sex=sex,
            duration=duration,
            percentrage=percentrage,
            mean_key=mean_key,
            extrapolation=extrapolation,
            NH_limit=NH_limit
        )

    def calculate_auditory_nihl(
        self,
        date: Optional[datetime] = None,
        freq_keys: List[str] = None,
        apply_correction: bool = False
    ) -> Dict[str, float]:
        """
        计算指定日期的听力 NIHL 值
        
        便捷方法，自动注入 staff_sex/staff_age 到 AuditoryHealthInfo.calculate_NIHL()
        
        Args:
            date: 体检日期，默认使用最近一次
            freq_keys: 频率键列表，如 ["1234", "346"]
            apply_correction: 是否应用年龄校正
            
        Returns:
            {"1234": float, "346": float}
        """
        if freq_keys is None:
            freq_keys = ["1234", "346"]
            
        # Get record key
        record_key = self.get_record_key(date)
        if record_key is None:
            raise ValueError("No health record found")
        
        health_info = self.staff_health_info.get(record_key)
        if health_info is None or health_info.auditory is None:
            raise ValueError(f"No auditory health info found for date {record_key}")
        
        # Get age for this record (adjust based on record date)
        age = self.staff_age
        if age is not None and isinstance(record_key, datetime):
            # Adjust age based on how many years have passed since first record
            first_date = min(self.record_dates) if self.record_dates else record_key
            years_diff = (record_key - first_date).days // 365
            # If we have time series, the age might need adjustment
            # This is a simplification; in practice, age should be calculated from birth date
        
        return health_info.auditory.calculate_NIHL(
            sex=self.staff_sex,
            age=age,
            freq_keys=freq_keys,
            apply_correction=apply_correction
        )

