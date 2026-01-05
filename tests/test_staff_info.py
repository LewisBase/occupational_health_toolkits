try:
    from ohtk.staff_info import StaffInfo
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from ohtk.staff_info import StaffInfo

test_data = {
    "staff_id": 1,
    "record_year": [2021, 2023],
    "name": ["na", "nb"],
    "factory_name": ["fa", "fb"],
    "work_shop": ["wsa", "wsb"],
    "work_position": ["wpa", "wpb"],
    "sex": ["F", "M"],
    "age": [22, 35],
    "duration": [5, 8],
    "smoking": ["y", "y"],
    "year_of_smoking": [2, 3],
    "cigaretee_per_day": [4, 5],
    "occupational_clinic_class": ["occa", "occb"],
    "auditory_detection": ["test","test"],
    "auditory_diagnose": ["test", "test"],
    "noise_hazard_info": ["test", "test"]
}

staff_info = StaffInfo(**test_data)