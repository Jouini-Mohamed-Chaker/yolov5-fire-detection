# config.py
import datetime
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set path and confidence threshold
MODEL_PATH = Path(__file__).resolve().parent.parent / "models/yolov5s_best.pt"
CONFIDENCE_THRESHOLD = 0.5

# Ensure the model file exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Initialize model device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Load YOLOv5 model locally
    model = torch.hub.load('yolov5', 'custom', path=str(MODEL_PATH), source='local')
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Error loading YOLOv5 model.")
    raise e
