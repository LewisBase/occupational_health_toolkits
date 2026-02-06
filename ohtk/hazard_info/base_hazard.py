from pydantic import BaseModel
from typing import List, Dict, Optional

class BaseHazard(BaseModel):
    recorder: Optional[str] = None
    recorder_time: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._build(**data)
        
    def _build(self, **data):
        # 设置记录器和记录时间
        self.recorder = data.get('recorder', self.recorder)
        self.recorder_time = data.get('recorder_time', self.recorder_time)