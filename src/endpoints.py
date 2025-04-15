# endpoints.py
import uuid
import time
import asyncio
import datetime
import threading
import cv2
import numpy as np
import logging
from fastapi import APIRouter, HTTPException, Response, UploadFile, File, Query, Path
from fastapi.responses import StreamingResponse, JSONResponse
from .models import StreamRequest
from .stream_manager import StreamManager
from .detection import detect_fire

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory store for active streams and locks
active_streams = {}
stream_locks = {}

@router.post("/add_stream", status_code=201)
async def add_stream(request: StreamRequest):
    camera_url = request.camera_url
    # Check for existing stream with the same URL
    for stream_id, manager in active_streams.items():
        if manager.camera_url == camera_url:
            return JSONResponse(
                status_code=409,
                content={
                    "stream_id": stream_id,
                    "stream_url": f"/stream/{stream_id}",
                    "message": "Stream with this URL already exists."
                }
            )
    stream_id = str(uuid.uuid4())
    manager = StreamManager(camera_url, stream_id, request.name)
    if not manager.start():
        raise HTTPException(status_code=500, detail="Failed to initialize stream")
    active_streams[stream_id] = manager
    stream_locks[stream_id] = threading.Lock()
    return {
        "stream_id": stream_id,
        "status": "active",
        "stream_url": f"/stream/{stream_id}",
        "created_at": manager.created_at,
        "message": "Stream successfully added."
    }

@router.get("/stream/{stream_id}")
async def stream_video(stream_id: str = Path(..., description="ID of the stream to access")):
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
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                await asyncio.sleep(0.033)
        finally:
            if stream_id in active_streams:
                active_streams[stream_id].remove_client(client_id)
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@router.delete("/delete_stream/{stream_id}")
async def delete_stream(stream_id: str = Path(..., description="ID of the stream to delete")):
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    with stream_locks[stream_id]:
        active_streams[stream_id].stop()
        del active_streams[stream_id]
        del stream_locks[stream_id]
    return {
        "stream_id": stream_id,
        "status": "deleted",
        "message": "Stream has been successfully deleted and resources released."
    }

@router.get("/list_streams")
async def list_streams():
    streams_info = [manager.get_status() for manager in active_streams.values()]
    return {
        "streams": streams_info,
        "total_count": len(streams_info)
    }

@router.get("/stream_status/{stream_id}")
async def stream_status(stream_id: str = Path(..., description="ID of the stream to check status")):
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    return active_streams[stream_id].get_status()

@router.post("/detect_frame")
async def detect_frame(
    file: UploadFile = File(...),
    frame_id: str = None,
    timestamp: str = None
):
    try:
        frame_id = frame_id or str(uuid.uuid4())
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
        results = detect_fire(frame)
        _, buffer = cv2.imencode('.jpg', results["annotated_frame"])
        headers = {
            "X-Frame-ID": frame_id,
            "X-Detections-Count": str(len(results["detections"])),
            "Content-Type": "image/jpeg"
        }
        return Response(content=buffer.tobytes(), headers=headers, media_type="image/jpeg")
    except Exception as e:
        logger.exception(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

# Background task to clean up idle streams
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_idle_streams())

async def cleanup_idle_streams():
    while True:
        try:
            await asyncio.sleep(60)
            stream_ids_to_delete = []
            current_time = time.time()
            for stream_id, manager in active_streams.items():
                if not manager.clients and current_time - manager.last_accessed > manager.idle_timeout:
                    stream_ids_to_delete.append(stream_id)
            for stream_id in stream_ids_to_delete:
                if stream_id in active_streams:
                    with stream_locks[stream_id]:
                        active_streams[stream_id].stop()
                        del active_streams[stream_id]
                        del stream_locks[stream_id]
                    logger.info(f"Auto-deleted idle stream: {stream_id}")
        except Exception as e:
            logger.exception(f"Error in cleanup task: {str(e)}")
