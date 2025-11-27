# app/schemas.py

from pydantic import BaseModel
from typing import Optional


class SepsisRequest(BaseModel):

    Hour: float
    HR: float
    O2Sat: float
    SBP: float
    MAP: float
    DBP: float
    Resp: float
    Age: float
    Gender: float
    Unit1: float
    Unit2: float
    HospAdmTime: float
    ICULOS: float

class SepsisResponse(BaseModel):
    probability: float
    predicted_label: int
    threshold_used: float
