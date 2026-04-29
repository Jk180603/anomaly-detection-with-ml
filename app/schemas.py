from pydantic import BaseModel
from typing import List


class SensorSequenceRequest(BaseModel):
    sequence: List[List[float]]