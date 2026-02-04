# -*- coding: utf-8 -*-
"""
Unit tests for ohtk refactoring - ChinaCDC data compatibility

Tests cover:
1. PTA correction module (pta_correction.py)
2. StaffBasicInfo with new fields (record-specific, no sex/age)
3. StaffHealthInfo with nested AuditoryHealthInfo
4. AuditoryHealthInfo for hearing check data
5. StaffInfo with datetime indexing and ChinaCDC data format
6. Model loader utilities
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


class TestPTACorrection:
    """Tests for PTA correction module."""
    
    def test_get_age_segment(self):
        """Test age segment calculation."""
        from ohtk.utils.pta_correction import get_age_segment
        
        assert get_age_segment(15) == "0-19"
        assert get_age_segment(25) == "20-29"
        assert get_age_segment(35) == "30-39"
        assert get_age_segment(45) == "40-49"
        assert get_age_segment(55) == "50-59"
        assert get_age_segment(65) == "60-69"
        assert get_age_segment(75) == "70-"
    
    def test_normalize_sex(self):
        """Test sex normalization."""
        from ohtk.utils.pta_correction import normalize_sex
        
        assert normalize_sex("Male") == "M"
        assert normalize_sex("男") == "M"
        assert normalize_sex("M") == "M"
        assert normalize_sex("m") == "M"
        assert normalize_sex("Female") == "F"
        assert normalize_sex("女") == "F"
        assert normalize_sex("F") == "F"
    
    def test_get_correction_value(self):
        """Test age correction value lookup."""
        from ohtk.utils.pta_correction import get_correction_value
        
        # Young age - no correction
        assert get_correction_value(25, "M", 4000) == 0
        
        # Middle age - has correction
        assert get_correction_value(45, "M", 4000) == 8
        assert get_correction_value(45, "F", 4000) == 4
        
        # Older age - higher correction
        assert get_correction_value(65, "M", 4000) == 28
        assert get_correction_value(65, "F", 4000) == 16
    
    def test_correct_pta_value(self):
        """Test PTA value correction."""
        from ohtk.utils.pta_correction import correct_pta_value
        
        # Raw value 40 dB, age 45, male, 4000 Hz
        corrected = correct_pta_value(40, 45, "M", 4000)
        assert corrected == 32  # 40 - 8
        
        # NaN handling
        assert np.isnan(correct_pta_value(np.nan, 45, "M", 4000))
    
    def test_calculate_nihl_346(self):
        """Test NIHL346 calculation (high frequency)."""
        from ohtk.utils.pta_correction import calculate_nihl
        
        ear_data = {
            "left_ear_3000": 25.0, "right_ear_3000": 30.0,
            "left_ear_4000": 35.0, "right_ear_4000": 40.0,
            "left_ear_6000": 45.0, "right_ear_6000": 50.0,
        }
        
        nihl = calculate_nihl(ear_data, freq_key="346")
        # Average of all 6 values
        expected = np.mean([25, 30, 35, 40, 45, 50])
        assert abs(nihl - expected) < 0.01
    
    def test_calculate_nihl_1234(self):
        """Test NIHL1234 calculation (speech frequency)."""
        from ohtk.utils.pta_correction import calculate_nihl
        
        ear_data = {
            "left_ear_1000": 10.0, "right_ear_1000": 15.0,
            "left_ear_2000": 20.0, "right_ear_2000": 25.0,
            "left_ear_3000": 30.0, "right_ear_3000": 35.0,
            "left_ear_4000": 40.0, "right_ear_4000": 45.0,
        }
        
        nihl = calculate_nihl(ear_data, freq_key="1234")
        expected = np.mean([10, 15, 20, 25, 30, 35, 40, 45])
        assert abs(nihl - expected) < 0.01
    
    def test_calculate_all_nihl(self):
        """Test calculating all NIHL indicators."""
        from ohtk.utils.pta_correction import calculate_all_nihl
        
        ear_data = {
            "left_ear_1000": 10.0, "right_ear_1000": 10.0,
            "left_ear_2000": 15.0, "right_ear_2000": 15.0,
            "left_ear_3000": 20.0, "right_ear_3000": 20.0,
            "left_ear_4000": 25.0, "right_ear_4000": 25.0,
            "left_ear_6000": 30.0, "right_ear_6000": 30.0,
        }
        
        result = calculate_all_nihl(ear_data)
        assert "1234" in result
        assert "346" in result
        assert result["1234"] == 17.5  # avg of 10,10,15,15,20,20,25,25
        assert result["346"] == 25.0   # avg of 20,20,25,25,30,30
    
    def test_classify_hearing_loss(self):
        """Test hearing loss classification."""
        from ohtk.utils.pta_correction import classify_hearing_loss
        
        assert classify_hearing_loss(20) == "正常"
        assert classify_hearing_loss(30) == "轻度"
        assert classify_hearing_loss(45) == "中度"
        assert classify_hearing_loss(60) == "重度"
        assert classify_hearing_loss(np.nan) == "未知"


class TestStaffBasicInfo:
    """Tests for StaffBasicInfo with new fields (record-specific, no sex/age)."""
    
    def test_basic_creation(self):
        """Test basic StaffBasicInfo creation with required fields."""
        from ohtk.staff_info import StaffBasicInfo
        
        info = StaffBasicInfo(
            staff_id="worker_001",
            creation_date=datetime(2023, 6, 15),
            creation_province="浙江",
        )
        
        assert info.staff_id == "worker_001"
        assert info.creation_province == "浙江"
        # sex/age should NOT be fields anymore
        assert not hasattr(info, 'sex') or info.model_fields.get('sex') is None
    
    def test_with_chinacdc_fields(self):
        """Test StaffBasicInfo with ChinaCDC-specific fields."""
        from ohtk.staff_info import StaffBasicInfo
        
        info = StaffBasicInfo(
            staff_id="worker_002",
            creation_date=datetime(2023, 6, 15),
            creation_province="浙江",
            creation_city="杭州市",
            industry_category="电气机械和器材制造业",
            examination_type_code=1,
        )
        
        assert info.creation_province == "浙江"
        assert info.creation_city == "杭州市"
        assert info.industry_category == "电气机械和器材制造业"
        assert info.examination_type_code == 1
    
    def test_optional_fields(self):
        """Test that traditional required fields are now optional."""
        from ohtk.staff_info import StaffBasicInfo
        
        # Should work without factory_name, work_shop, work_position
        info = StaffBasicInfo(
            staff_id="worker_003",
            creation_date=datetime(2023, 1, 1),
            creation_province="未知",
        )
        
        assert info.staff_id == "worker_003"
        assert info.factory_name is None
        assert info.work_shop is None
        assert info.work_position is None


class TestAuditoryHealthInfo:
    """Tests for AuditoryHealthInfo (nested under StaffHealthInfo)."""
    
    def test_direct_nihl_input(self):
        """Test directly providing NIHL values."""
        from ohtk.staff_info import AuditoryHealthInfo
        
        info = AuditoryHealthInfo(
            NIHL={"1234": 25.5, "346": 32.0},
        )
        
        assert info.NIHL is not None
        assert info.NIHL.get("1234") == 25.5
        assert info.NIHL.get("346") == 32.0
    
    def test_nihl_with_classification(self):
        """Test NIHL with hearing loss classification."""
        from ohtk.staff_info import AuditoryHealthInfo
        
        info = AuditoryHealthInfo(
            NIHL={"1234": 20.0, "346": 35.0},
        )
        
        assert info.norm_hearing_loss == "正常"
        assert info.high_hearing_loss == "轻度"
    
    def test_nihl_calculation_from_ear_data(self):
        """Test NIHL calculation from ear data."""
        from ohtk.staff_info import AuditoryHealthInfo
        
        ear_data = {
            "left_ear_3000": 25.0, "right_ear_3000": 25.0,
            "left_ear_4000": 30.0, "right_ear_4000": 30.0,
            "left_ear_6000": 35.0, "right_ear_6000": 35.0,
        }
        
        info = AuditoryHealthInfo(ear_data=ear_data)
        
        # sex/age must be passed explicitly (from StaffInfo)
        nihl = info.calculate_NIHL(sex="M", age=45, freq_keys=["346"])
        assert "346" in nihl
        assert nihl["346"] == 30.0
    
    def test_get_nihl(self):
        """Test get_nihl helper method."""
        from ohtk.staff_info import AuditoryHealthInfo
        
        info = AuditoryHealthInfo(
            NIHL={"1234": 22.0, "346": 28.0},
        )
        
        assert info.get_nihl("346") == 28.0
        assert info.get_nihl("1234") == 22.0
        assert info.get_nihl("unknown") is None


class TestStaffHealthInfo:
    """Tests for StaffHealthInfo with nested AuditoryHealthInfo."""
    
    def test_nested_auditory_info(self):
        """Test StaffHealthInfo with nested AuditoryHealthInfo."""
        from ohtk.staff_info import StaffHealthInfo, AuditoryHealthInfo
        
        auditory = AuditoryHealthInfo(
            NIHL={"1234": 25.5, "346": 32.0},
        )
        
        info = StaffHealthInfo(
            staff_id="worker_001",
            auditory=auditory,
        )
        
        assert info.auditory is not None
        assert info.auditory.NIHL.get("1234") == 25.5
        assert info.auditory.NIHL.get("346") == 32.0
        assert "auditory" in info.diagnose_types
    
    def test_convenience_methods(self):
        """Test convenience methods for accessing auditory data."""
        from ohtk.staff_info import StaffHealthInfo, AuditoryHealthInfo
        
        auditory = AuditoryHealthInfo(
            NIHL={"1234": 22.0, "346": 28.0},
        )
        
        info = StaffHealthInfo(
            staff_id="worker_002",
            auditory=auditory,
        )
        
        assert info.get_auditory_nihl("346") == 28.0
        assert info.get_auditory_nihl("1234") == 22.0
        assert info.has_auditory_data() is True
    
    def test_without_auditory(self):
        """Test StaffHealthInfo without auditory data."""
        from ohtk.staff_info import StaffHealthInfo
        
        info = StaffHealthInfo(
            staff_id="worker_003",
        )
        
        assert info.auditory is None
        assert info.has_auditory_data() is False
        assert info.get_auditory_nihl("346") is None


class TestStaffInfo:
    """Tests for StaffInfo with datetime indexing and ChinaCDC support."""
    
    def test_chinacdc_single_record(self):
        """Test loading single ChinaCDC format record."""
        from ohtk.staff_info import StaffInfo
        
        data = {
            "staff_id": "worker_001",
            "creation_date": "2023-06-15",
            "sex": "M",
            "age": 40,
            "NIHL1234": 22.5,
            "NIHL346": 30.0,
            "LEX_8h_LEX_40h_median": 85.5,
            "creation_province": "浙江",
            "creation_city": "杭州市",
            "industry_category": "电气机械和器材制造业",
            "check_order": 1,
            "days_since_first": 0,
        }
        
        staff = StaffInfo(**data)
        
        assert staff.staff_id == "worker_001"
        assert staff.staff_sex == "M"
        assert staff.staff_age == 40
        assert len(staff.record_dates) == 1
        
        # Check basic info
        date_key = staff.record_dates[0]
        basic_info = staff.staff_basic_info[date_key]
        assert basic_info.creation_province == "浙江"
        assert basic_info.creation_city == "杭州市"
        
        # Check health info (nested auditory)
        health_info = staff.staff_health_info[date_key]
        assert health_info.auditory is not None
        assert health_info.auditory.NIHL["1234"] == 22.5
        assert health_info.auditory.NIHL["346"] == 30.0
        
        # Check occupational hazard info
        occhaz_info = staff.staff_occhaz_info[date_key]
        assert occhaz_info.noise_hazard_info.LAeq == 85.5
    
    def test_chinacdc_multiple_records(self):
        """Test loading multiple ChinaCDC format records."""
        from ohtk.staff_info import StaffInfo
        
        data = {
            "staff_id": "worker_002",
            "creation_date": ["2021-03-15", "2022-06-20", "2023-09-10"],
            "sex": ["M", "M", "M"],
            "age": [38, 39, 40],
            "NIHL1234": [20.0, 22.0, 24.0],
            "NIHL346": [25.0, 28.0, 31.0],
            "LEX_8h_LEX_40h_median": [85.0, 86.0, 87.0],
            "creation_province": ["浙江", "浙江", "浙江"],
        }
        
        staff = StaffInfo(**data)
        
        assert len(staff.record_dates) == 3
        assert len(staff.check_order) == 3
        assert len(staff.days_since_first) == 3
        
        # Check time-series features
        first_date = min(staff.record_dates)
        assert staff.check_order[first_date] == 1
        assert staff.days_since_first[first_date] == 0
    
    def test_load_from_dataframe(self):
        """Test loading StaffInfo from DataFrame."""
        from ohtk.staff_info import StaffInfo
        
        df = pd.DataFrame({
            "worker_id": ["w1", "w1", "w1", "w2", "w2"],
            "creation_date": ["2021-01-15", "2022-03-20", "2023-05-25", "2022-01-01", "2023-01-01"],
            "sex": ["M", "M", "M", "F", "F"],
            "age": [35, 36, 37, 40, 41],
            "NIHL346": [25.0, 27.0, 29.0, 30.0, 32.0],
            "LEX_8h_LEX_40h_median": [85.0, 86.0, 87.0, 82.0, 83.0],
            "creation_province": ["浙江", "浙江", "浙江", "江苏", "江苏"],
        })
        
        staff = StaffInfo.load_from_dataframe(df, "w1")
        
        assert staff.staff_id == "w1"
        assert len(staff.record_dates) == 3
        assert staff.staff_sex == "M"
    
    def test_load_batch_from_dataframe(self):
        """Test batch loading multiple workers from DataFrame."""
        from ohtk.staff_info import StaffInfo
        
        df = pd.DataFrame({
            "worker_id": ["w1", "w1", "w2", "w2", "w3"],
            "creation_date": ["2021-01-01", "2022-01-01", "2021-06-01", "2022-06-01", "2023-01-01"],
            "sex": ["M", "M", "F", "F", "M"],
            "age": [30, 31, 40, 41, 50],
            "NIHL346": [20.0, 22.0, 25.0, 27.0, 35.0],
            "creation_province": ["浙江", "浙江", "江苏", "江苏", "广东"],
        })
        
        result = StaffInfo.load_batch_from_dataframe(df)
        
        assert "w1" in result
        assert "w2" in result
        assert "w3" in result
        assert len(result["w1"].record_dates) == 2
        assert len(result["w2"].record_dates) == 2
        assert len(result["w3"].record_dates) == 1
    
    def test_get_most_recent_date(self):
        """Test getting most recent record date."""
        from ohtk.staff_info import StaffInfo
        
        data = {
            "staff_id": "worker_003",
            "creation_date": ["2021-01-01", "2023-06-15", "2022-03-20"],
            "sex": ["M", "M", "M"],
            "age": [35, 37, 36],
            "creation_province": ["浙江", "浙江", "浙江"],
        }
        
        staff = StaffInfo(**data)
        most_recent = staff.get_most_recent_date()
        
        assert most_recent.year == 2023
        assert most_recent.month == 6
    
    def test_backward_compatibility_year_format(self):
        """Test backward compatibility with year-based format."""
        from ohtk.staff_info import StaffInfo
        
        # This format should still work (legacy format)
        data = {
            "staff_id": "worker_004",
            "record_year": [2021, 2022, 2023],
            "name": ["张三", "张三", "张三"],
            "factory_name": ["工厂A", "工厂A", "工厂A"],
            "work_shop": ["车间1", "车间1", "车间1"],
            "work_position": ["操作工", "操作工", "操作工"],
            "sex": ["M", "M", "M"],
            "age": [35, 36, 37],
            "duration": [5, 6, 7],
        }
        
        staff = StaffInfo(**data)
        
        assert staff.staff_id == "worker_004"
        assert staff.staff_name == "张三"
        assert len(staff.record_years) == 3
    
    def test_attribute_transmission(self):
        """Test that sex/age/duration are stored only in StaffInfo."""
        from ohtk.staff_info import StaffInfo
        
        data = {
            "staff_id": "worker_005",
            "creation_date": "2023-06-15",
            "sex": "M",
            "age": 45,
            "duration": 10.5,
            "NIHL1234": 25.0,
            "NIHL346": 30.0,
            "creation_province": "浙江",
        }
        
        staff = StaffInfo(**data)
        
        # sex/age/duration should be in StaffInfo
        assert staff.staff_sex == "M"
        assert staff.staff_age == 45
        assert staff.staff_duration == 10.5
        
        # StaffBasicInfo should NOT have sex/age/duration
        date_key = staff.record_dates[0]
        basic_info = staff.staff_basic_info[date_key]
        assert not hasattr(basic_info, 'sex') or getattr(basic_info, 'sex', None) is None


class TestAuditoryDiagnose:
    """Tests for AuditoryDiagnose static methods."""
    
    def test_nihl_static_method(self):
        """Test NIHL calculation via AuditoryDiagnose."""
        from ohtk.diagnose_info import AuditoryDiagnose
        
        ear_data = {
            "left_ear_3000": 25.0, "right_ear_3000": 30.0,
            "left_ear_4000": 35.0, "right_ear_4000": 40.0,
            "left_ear_6000": 45.0, "right_ear_6000": 50.0,
        }
        
        nihl = AuditoryDiagnose.NIHL(ear_data, freq_key="346")
        expected = np.mean([25, 30, 35, 40, 45, 50])
        assert abs(nihl - expected) < 0.01
    
    def test_nihl_all_static_method(self):
        """Test NIHL_all calculation via AuditoryDiagnose."""
        from ohtk.diagnose_info import AuditoryDiagnose
        
        ear_data = {
            "left_ear_1000": 10.0, "right_ear_1000": 10.0,
            "left_ear_2000": 15.0, "right_ear_2000": 15.0,
            "left_ear_3000": 20.0, "right_ear_3000": 20.0,
            "left_ear_4000": 25.0, "right_ear_4000": 25.0,
            "left_ear_6000": 30.0, "right_ear_6000": 30.0,
        }
        
        result = AuditoryDiagnose.NIHL_all(ear_data)
        assert "1234" in result
        assert "346" in result


class TestModelLoader:
    """Tests for model loader utilities."""
    
    def test_list_available_models(self):
        """Test listing available models."""
        from ohtk.staff_info.models import ModelLoader
        
        # Test with a non-existent directory
        result = ModelLoader.list_available_models("/non/existent/path")
        assert result == {}
    
    def test_get_model_info(self):
        """Test parsing model info from filename."""
        from ohtk.staff_info.models import ModelLoader
        
        info = ModelLoader.get_model_info("32-hearing_loss-province-model.pkl")
        
        assert info["region_code"] == "32"
        assert info["target"] == "hearing_loss"
        assert info["scope"] == "province"


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_chinacdc_workflow(self):
        """Test complete ChinaCDC data loading workflow."""
        from ohtk.staff_info import StaffInfo
        
        # Simulate ChinaCDC queue data
        df = pd.DataFrame({
            "worker_id": ["w1"] * 4,
            "creation_date": pd.date_range("2021-01-01", periods=4, freq="6ME"),
            "sex": ["M"] * 4,
            "age": [40, 40, 41, 41],
            "NIHL1234": [20.0, 22.0, 24.0, 26.0],
            "NIHL346": [25.0, 28.0, 31.0, 34.0],
            "LEX_8h_LEX_40h_median": [85.0, 86.0, 87.0, 88.0],
            "creation_province": ["浙江"] * 4,
            "creation_city": ["杭州市"] * 4,
            "check_order": [1, 2, 3, 4],
            "days_since_first": [0, 183, 365, 548],
        })
        
        # Load worker
        staff = StaffInfo.load_from_dataframe(df, "w1")
        
        # Verify data structure
        assert len(staff.record_dates) == 4
        assert staff.staff_sex == "M"
        
        # Verify time-series features
        for date in staff.record_dates:
            assert date in staff.check_order
            assert date in staff.days_since_first
        
        # Verify NIHL progression (via nested auditory info)
        nihl_values = []
        for date in sorted(staff.record_dates):
            health_info = staff.staff_health_info[date]
            if health_info.auditory and health_info.auditory.NIHL:
                nihl_values.append(health_info.auditory.NIHL.get("346"))
        
        # NIHL should be increasing
        assert nihl_values == sorted(nihl_values)
    
    def test_pta_correction_integration(self):
        """Test PTA correction integrated with AuditoryHealthInfo."""
        from ohtk.staff_info import AuditoryHealthInfo
        from ohtk.utils.pta_correction import calculate_all_nihl
        
        # Create ear data
        ear_data = {
            "left_ear_1000": 15.0, "right_ear_1000": 15.0,
            "left_ear_2000": 20.0, "right_ear_2000": 20.0,
            "left_ear_3000": 25.0, "right_ear_3000": 25.0,
            "left_ear_4000": 30.0, "right_ear_4000": 30.0,
            "left_ear_6000": 35.0, "right_ear_6000": 35.0,
        }
        
        # Calculate NIHL using utility
        nihl_result = calculate_all_nihl(ear_data)
        
        # Create auditory health info with ear data
        auditory_info = AuditoryHealthInfo(ear_data=ear_data)
        
        # Calculate NIHL via auditory info (need to pass sex/age explicitly)
        auditory_nihl = auditory_info.calculate_NIHL(sex="M", age=45)
        
        # Results should match
        assert abs(nihl_result["1234"] - auditory_nihl["1234"]) < 0.01
        assert abs(nihl_result["346"] - auditory_nihl["346"]) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
