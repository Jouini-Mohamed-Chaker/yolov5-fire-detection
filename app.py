import os
import uuid
import time
import logging
import base64
import threading
import asyncio
import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body, Query, Path, Request, Response, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import uvicorn
from io import BytesIO
from PIL import Image
import datetime
import json
from pathlib import Path as FilePath

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load YOLOv5 model
MODEL_PATH = FilePath('models/yolov5s_best.pt')
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Initialize YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('yolov5', 'custom', path=MODEL_PATH, source='local')
model.to(device)
model.eval()

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Pydantic models for API
class StreamRequest(BaseModel):
    camera_url: str
    name: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class Detection(BaseModel):
    label: str
    confidence: float
    bounding_box: Dict[str, float]

# Update DetectionResponse to only return metadata (optional use)
class DetectionResponse(BaseModel):
    frame_id: str
    detections: List[Detection]



# Create FastAPI app
app = FastAPI(
    title="Fire Detection System",
    description="Real-time fire detection system using YOLOv5",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Store active streams
active_streams = {}
stream_locks = {}  # To prevent race conditions when accessing streams

# Class to manage video streams
class StreamManager:
    def __init__(self, camera_url, stream_id, name=None):
        self.camera_url = camera_url
        self.stream_id = stream_id
        self.name = name or f"Stream {stream_id[:8]}"
        self.cap = None
        self.running = False
        self.last_accessed = time.time()
        self.created_at = datetime.datetime.now().isoformat()
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        self.clients = set()
        self.idle_timeout = 300  # 5 minutes
        self.total_detections = 0
        self.last_detection_time = None
        self.current_fps = 0
        self.resolution = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3

    def start(self):
        if self.running:
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_url)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera stream: {self.camera_url}")
            
            # Get stream info
            self.resolution = f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_frames)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            logger.info(f"Started stream {self.stream_id} from {self.camera_url}")
            return True
        except Exception as e:
            logger.error(f"Error starting stream {self.stream_id}: {str(e)}")
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def stop(self):
        if not self.running:
            return
        
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info(f"Stopped stream {self.stream_id}")

    def _process_frames(self):
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            # Check if stream is idle and no clients are connected
            if not self.clients and time.time() - self.last_accessed > self.idle_timeout:
                logger.info(f"Stream {self.stream_id} idle for {self.idle_timeout} seconds, stopping")
                self.stop()
                break
                
            # Read frame
            success, frame = self.cap.read()
            if not success:
                logger.warning(f"Failed to read frame from {self.camera_url}")
                # Try to reconnect
                self.reconnect_attempts += 1
                if self.reconnect_attempts <= self.max_reconnect_attempts:
                    logger.info(f"Attempting reconnect {self.reconnect_attempts}/{self.max_reconnect_attempts}")
                    time.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
                    
                    # Try reopening the capture
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(self.camera_url)
                    if self.cap.isOpened():
                        self.reconnect_attempts = 0  # Reset on success
                        continue
                else:
                    logger.error(f"Failed to reconnect to {self.camera_url} after {self.max_reconnect_attempts} attempts, stopping stream")
                    self.stop()
                    break
            else:
                self.reconnect_attempts = 0  # Reset reconnect attempts counter on successful frame read
            
            # Process frame through detection model
            results = detect_fire(frame)
            annotated_frame = results["annotated_frame"]
            detections = results["detections"]
            
            # Update detection stats
            if detections:
                self.total_detections += len(detections)
                self.last_detection_time = datetime.datetime.now().isoformat()
            
            # Calculate current FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:  # Update FPS every second
                self.current_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Update frame buffer
            with self.frame_lock:
                self.frame_buffer = annotated_frame
                self.last_accessed = time.time()
            
            # Sleep to control frame rate (adjust as needed)
            time.sleep(0.033)  # ~30 FPS
    
    def get_frame(self):
        """Get the latest processed frame"""
        with self.frame_lock:
            if self.frame_buffer is None:
                # Return black frame if no frames are available yet
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()
            
            self.last_accessed = time.time()
            _, buffer = cv2.imencode('.jpg', self.frame_buffer)
            return buffer.tobytes()
    
    def add_client(self, client_id):
        self.clients.add(client_id)
        logger.info(f"Added client {client_id} to stream {self.stream_id}")
    
    def remove_client(self, client_id):
        if client_id in self.clients:
            self.clients.remove(client_id)
            logger.info(f"Removed client {client_id} from stream {self.stream_id}")

    def get_status(self):
        """Get current status information for the stream"""
        uptime = 0
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            uptime = int(time.time() - self.last_accessed + self.idle_timeout)
            
        return {
            "stream_id": self.stream_id,
            "name": self.name,
            "url": self.camera_url,
            "status": "active" if self.running else "inactive",
            "fps": round(self.current_fps, 2),
            "resolution": self.resolution or "unknown",
            "uptime": uptime,
            "detections_count": self.total_detections,
            "last_detection": self.last_detection_time,
            "client_count": len(self.clients),
            "created_at": self.created_at,
            "last_activity": datetime.datetime.fromtimestamp(self.last_accessed).isoformat()
        }

# Function to detect fire in a frame
def detect_fire(frame):
    # Convert to RGB for model inference
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(frame_rgb)
    
    # Process results
    predictions = results.pandas().xyxy[0]
    detections = []
    
    # Create a copy of the frame for annotations
    annotated_frame = frame.copy()
    
    # Process each detection
    for _, prediction in predictions.iterrows():
        confidence = prediction['confidence']
        
        # Filter by confidence threshold
        if confidence >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, [prediction['xmin'], prediction['ymin'], prediction['xmax'], prediction['ymax']])
            label = prediction['name']
            
            # Calculate area percentage
            frame_area = annotated_frame.shape[0] * annotated_frame.shape[1]
            detection_area = (x2 - x1) * (y2 - y1)
            area_percentage = (detection_area / frame_area) * 100
            
            # Draw bounding box (red for high confidence, orange for lower)
            box_color = (0, 0, 255) if confidence > 0.8 else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Create semi-transparent background for text
            text_size = cv2.getTextSize(f"{label}: {confidence:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), box_color, -1)
            
            # Add label and confidence with white text
            text = f"{label}: {confidence:.2f}"
            cv2.putText(annotated_frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add to detections list
            detections.append({
                "label": label,
                "confidence": float(confidence),
                "bounding_box": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1)
                },
                "area_percentage": float(area_percentage)
            })
    
    # Add timestamp to the frame
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, timestamp, 
                (annotated_frame.shape[1] - 200, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return {
        "annotated_frame": annotated_frame,
        "detections": detections
    }

# Helper function to convert frame to base64
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Helper function to convert base64 to frame
def base64_to_frame(base64_str):
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Endpoint to add a new stream
@app.post("/add_stream", status_code=201)
async def add_stream(request: StreamRequest):
    camera_url = request.camera_url
    
    # Check if stream with this URL already exists
    for stream_id, manager in active_streams.items():
        if manager.camera_url == camera_url:
            return JSONResponse(
                status_code=409,  # Conflict
                content={
                    "stream_id": stream_id,
                    "stream_url": f"/stream/{stream_id}",
                    "message": "Stream with this URL already exists."
                }
            )
    
    # Create a new stream
    stream_id = str(uuid.uuid4())
    
    # Initialize stream manager
    manager = StreamManager(camera_url, stream_id, request.name)
    
    # Start the stream
    if not manager.start():
        raise HTTPException(status_code=500, detail="Failed to initialize stream")
    
    # Store the stream manager
    active_streams[stream_id] = manager
    stream_locks[stream_id] = threading.Lock()
    
    # Return stream details
    return {
        "stream_id": stream_id,
        "status": "active",
        "stream_url": f"/stream/{stream_id}",
        "created_at": manager.created_at,
        "message": "Stream successfully added."
    }

# Endpoint to stream a processed video
@app.get("/stream/{stream_id}")
async def stream_video(stream_id: str = Path(..., description="ID of the stream to access")):
    # Check if stream exists
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    manager = active_streams[stream_id]
    client_id = str(uuid.uuid4())
    manager.add_client(client_id)
    
    async def generate_frames():
        try:
            while stream_id in active_streams and active_streams[stream_id].running:
                frame_bytes = manager.get_frame()
                if frame_bytes:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
                await asyncio.sleep(0.033)  # ~30 FPS
        finally:
            # Clean up when client disconnects
            if stream_id in active_streams:
                active_streams[stream_id].remove_client(client_id)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"  # Disable proxy buffering for Nginx
        }
    )

# Endpoint to delete a stream
@app.delete("/delete_stream/{stream_id}")
async def delete_stream(stream_id: str = Path(..., description="ID of the stream to delete")):
    # Check if stream exists
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    with stream_locks[stream_id]:
        # Stop the stream
        active_streams[stream_id].stop()
        
        # Remove from active streams
        del active_streams[stream_id]
        del stream_locks[stream_id]
    
    return {
        "stream_id": stream_id,
        "status": "deleted",
        "message": "Stream has been successfully deleted and resources released."
    }

# Endpoint to list all active streams
@app.get("/list_streams")
async def list_streams():
    streams_info = []
    
    for stream_id, manager in active_streams.items():
        streams_info.append(manager.get_status())
    
    return {
        "streams": streams_info,
        "total_count": len(streams_info)
    }

# Endpoint for stream status
@app.get("/stream_status/{stream_id}")
async def stream_status(stream_id: str = Path(..., description="ID of the stream to check status")):
    # Check if stream exists
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    return active_streams[stream_id].get_status()

# Endpoint for independent frame detection
@app.post("/detect_frame")
async def detect_frame(
    file: UploadFile = File(...),
    frame_id: Optional[str] = None,
    timestamp: Optional[str] = None
):
    try:
        # Generate a frame ID if not provided
        frame_id = frame_id or str(uuid.uuid4())

        # Read the uploaded image as binary
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Run detection
        results = detect_fire(frame)

        # Encode annotated frame
        _, buffer = cv2.imencode('.jpg', results["annotated_frame"])

        # Create headers with detection info
        headers = {
            "X-Frame-ID": frame_id,
            "X-Detections-Count": str(len(results["detections"])),
            "Content-Type": "image/jpeg"
        }

        # Return raw image with metadata headers
        return Response(content=buffer.tobytes(), headers=headers, media_type="image/jpeg")

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

# Background task to clean up idle streams
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_idle_streams())

async def cleanup_idle_streams():
    while True:
        try:
            # Sleep for a while before checking
            await asyncio.sleep(60)  # Check every minute
            
            stream_ids_to_delete = []
            current_time = time.time()
            
            # Find idle streams
            for stream_id, manager in active_streams.items():
                if not manager.clients and current_time - manager.last_accessed > manager.idle_timeout:
                    stream_ids_to_delete.append(stream_id)
            
            # Delete idle streams
            for stream_id in stream_ids_to_delete:
                if stream_id in active_streams:
                    with stream_locks[stream_id]:
                        active_streams[stream_id].stop()
                        del active_streams[stream_id]
                        del stream_locks[stream_id]
                    logger.info(f"Auto-deleted idle stream: {stream_id}")
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8026, reload=True)