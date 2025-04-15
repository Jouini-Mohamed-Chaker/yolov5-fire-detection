# stream_manager.py
import cv2
import datetime
import time
import threading
import numpy as np
import logging
from .detection import detect_fire

logger = logging.getLogger(__name__)

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
        self.idle_timeout = 300  # seconds
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

            self.resolution = f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.processing_thread.start()
            logger.info(f"Started stream {self.stream_id} from {self.camera_url}")
            return True
        except Exception as e:
            logger.exception(f"Error starting stream {self.stream_id}: {str(e)}")
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
            # Stop stream if idle (no clients) for too long
            if not self.clients and time.time() - self.last_accessed > self.idle_timeout:
                logger.info(f"Stream {self.stream_id} idle for {self.idle_timeout} seconds, stopping")
                self.stop()
                break

            success, frame = self.cap.read()
            if not success:
                logger.warning(f"Failed to read frame from {self.camera_url}")
                self.reconnect_attempts += 1
                if self.reconnect_attempts <= self.max_reconnect_attempts:
                    logger.info(f"Reconnect attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} for stream {self.stream_id}")
                    time.sleep(2 ** self.reconnect_attempts)
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(self.camera_url)
                    if self.cap.isOpened():
                        self.reconnect_attempts = 0
                        continue
                else:
                    logger.error(f"Failed to reconnect after {self.max_reconnect_attempts} attempts for {self.camera_url}, stopping stream")
                    self.stop()
                    break
            else:
                self.reconnect_attempts = 0

            try:
                results = detect_fire(frame)
            except Exception as e:
                logger.exception(f"Detection failed on stream {self.stream_id}: {e}")
                continue

            annotated_frame = results["annotated_frame"]
            detections = results["detections"]

            if detections:
                self.total_detections += len(detections)
                self.last_detection_time = datetime.datetime.now().isoformat()

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                self.current_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            with self.frame_lock:
                self.frame_buffer = annotated_frame
                self.last_accessed = time.time()

            time.sleep(0.033)  # roughly 30 FPS

    def get_frame(self):
        with self.frame_lock:
            if self.frame_buffer is None:
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
