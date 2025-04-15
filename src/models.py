# models.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class StreamRequest(BaseModel):
    camera_url: str
    name: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class Detection(BaseModel):
    label: str
    confidence: float
    bounding_box: Dict[str, float]

class DetectionResponse(BaseModel):
    frame_id: str
    detections: List[Detection]
