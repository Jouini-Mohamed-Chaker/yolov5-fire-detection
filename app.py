from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import uuid
import torch
from pathlib import Path

# Configuration
MODEL_PATH = Path(__file__).parent / "models/yolov5s_best.pt"
CONFIDENCE_THRESHOLD = 0.5

# Load YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('yolov5', 'custom', path=str(MODEL_PATH), source='local')
model.to(device).eval()

# FastAPI app
app = FastAPI(title="Minimal Fire Detection Service", version="1.0.0")

# Pydantic schemas
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Detection(BaseModel):
    label: str
    confidence: float
    bounding_box: BoundingBox

class DetectionResponse(BaseModel):
    frame_id: str
    detections: list[Detection]

@app.post("/detect_frame", response_model=DetectionResponse)
async def detect_frame(file: UploadFile = File(...)):
    # Generate a unique frame ID
    frame_id = str(uuid.uuid4())

    # Read and decode image
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Convert to RGB and run model
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)
    preds = results.pandas().xyxy[0]

    # Collect detections
    detections = []
    h, w = frame.shape[:2]
    for _, row in preds.iterrows():
        conf = float(row.confidence)
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        detections.append(Detection(
            label=row.name,
            confidence=conf,
            bounding_box=BoundingBox(
                x=float(x1), y=float(y1),
                width=float(x2-x1), height=float(y2-y1)
            )
        ))

    return DetectionResponse(frame_id=frame_id, detections=detections)