from ohtk.staff_info.staff_info import StaffInfo
from ohtk.staff_info.staff_basic_info import StaffBasicInfo
from ohtk.staff_info.staff_health_info import StaffHealthInfo
from ohtk.staff_info.staff_occhaz_info import StaffOccHazInfo
from ohtk.staff_info.auditory_health_info import AuditoryHealthInfo

StaffInfo = StaffInfo
StaffBasicInfo = StaffBasicInfo
StaffStaffHealthInfo = None  # legacy alias placeholder
StaffHealthInfo = StaffHealthInfo
StaffOccHazInfo = StaffOccHazInfo
AuditoryHealthInfo = AuditoryHealthInfo
# ModelLoader/TimeSeriesPredictor/QueueModelPredictor removed in refactor(auditory) (commit caaf58b)
