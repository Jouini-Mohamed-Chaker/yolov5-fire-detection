import base64
import cv2
import numpy as np
import torch
import os
import logging
import time
from flask import Flask, request, jsonify
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

# Configuration (can be overridden with environment variables)
config = {
    "MODEL_PATH": os.environ.get("MODEL_PATH", "models/yolov5s_best.pt"),
    "MAX_WORKERS": int(os.environ.get("MAX_WORKERS", 4)),
    "BATCH_SIZE": int(os.environ.get("BATCH_SIZE", 4)),
    "QUEUE_SIZE": int(os.environ.get("QUEUE_SIZE", 32)),
    "PORT": int(os.environ.get("PORT", 5000)),
    "HOST": os.environ.get("HOST", "0.0.0.0"),
    "DEBUG": os.environ.get("DEBUG", "false").lower() == "true",
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
    "CONFIDENCE_THRESHOLD": float(os.environ.get("CONFIDENCE_THRESHOLD", 0.25)),
    "USE_GPU": os.environ.get("USE_GPU", "auto"),
}

# Configure logging
logging.basicConfig(
    level=getattr(logging, config["LOG_LEVEL"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fire_detection")

# Initialize Flask app
app = Flask(__name__)

# Request processing queue and response dictionary
request_queue = queue.Queue(maxsize=config["QUEUE_SIZE"])
response_dict = {}
response_lock = threading.Lock()

# Determine device
if config["USE_GPU"] == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cuda" if config["USE_GPU"].lower() == "true" and torch.cuda.is_available() else "cpu"

logger.info(f"Using device: {device}")

# Load the model
logger.info(f"Loading model from {config['MODEL_PATH']}")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=config["MODEL_PATH"])
model.to(device)
model.eval()
model.conf = config["CONFIDENCE_THRESHOLD"]
logger.info("Model loaded successfully")

def decode_frame(frame_b64):
    """Decode a base64-encoded image string to an RGB image."""
    try:
        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Decoded image is None")
        # Convert from BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

def batch_processor():
    """Process batches of images for inference"""
    batch = []
    batch_requests = []
    last_process_time = time.time()
    
    while True:
        try:
            # Get request from queue with timeout
            try:
                request_id, image = request_queue.get(timeout=0.1)
                batch.append(image)
                batch_requests.append(request_id)
                request_queue.task_done()
            except queue.Empty:
                pass
            
            current_time = time.time()
            should_process = False
            
            # Process if batch is full
            if len(batch) >= config["BATCH_SIZE"]:
                should_process = True
            # Or if we have items and waited too long
            elif batch and (current_time - last_process_time) > 0.2:
                should_process = True
                
            if should_process and batch:
                process_batch(batch, batch_requests)
                batch = []
                batch_requests = []
                last_process_time = current_time
                
        except Exception as e:
            logger.error(f"Error in batch processor: {e}")
            # If there was an error, set error response for all requests in batch
            with response_lock:
                for req_id in batch_requests:
                    response_dict[req_id] = {"error": str(e)}
            
            batch = []
            batch_requests = []

def process_batch(images, request_ids):
    """Run inference on a batch of images"""
    try:
        # Run inference
        with torch.no_grad():
            results = model(images)
        
        # Process results
        with response_lock:
            for i, request_id in enumerate(request_ids):
                # Extract detection results
                detections = results.xyxy[i].cpu().numpy().tolist()
                
                # Format detections as [x1, y1, x2, y2, confidence, class_id]
                formatted_detections = []
                for det in detections:
                    formatted_detections.append({
                        "bbox": det[:4],
                        "confidence": float(det[4]),
                        "class_id": int(det[5])
                    })
                
                # Store response
                response_dict[request_id] = {
                    "detections": formatted_detections,
                    "timestamp": time.time()
                }
        
        logger.debug(f"Processed batch of {len(images)} images")
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        # Set error response for all requests
        with response_lock:
            for request_id in request_ids:
                response_dict[request_id] = {"error": str(e)}

def cleanup_old_responses():
    """Clean up old responses periodically"""
    while True:
        time.sleep(60)  # Run every minute
        current_time = time.time()
        with response_lock:
            # Remove responses older than 5 minutes
            keys_to_remove = [
                k for k, v in response_dict.items() 
                if current_time - v.get("timestamp", 0) > 300
            ]
            for k in keys_to_remove:
                del response_dict[k]
        
        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} old responses")

@app.route('/detect', methods=['POST'])
def detect():
    """Handle frame detection request"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame provided in the request"}), 400
        
        # Generate unique request ID
        request_id = f"{time.time()}-{threading.get_ident()}"
        
        try:
            # Decode frame
            frame = decode_frame(data['frame'])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        # Add to queue
        try:
            request_queue.put((request_id, frame), block=False)
        except queue.Full:
            return jsonify({"error": "Service overloaded, please try again later"}), 503
        
        # Wait for response with timeout
        start_time = time.time()
        max_wait = 10  # 10 seconds timeout
        
        while time.time() - start_time < max_wait:
            with response_lock:
                if request_id in response_dict:
                    response = response_dict[request_id]
                    # Clean up
                    del response_dict[request_id]
                    
                    if "error" in response:
                        return jsonify({"error": response["error"]}), 500
                    
                    return jsonify(response), 200
            
            # Short sleep to prevent CPU spinning
            time.sleep(0.05)
        
        return jsonify({"error": "Request timed out"}), 408
        
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with service stats"""
    stats = {
        "status": "fire detection service is running",
        "device": device,
        "queue_usage": f"{request_queue.qsize()}/{config['QUEUE_SIZE']}",
        "workers": config["MAX_WORKERS"],
        "batch_size": config["BATCH_SIZE"],
        "pending_responses": len(response_dict)
    }
    
    return jsonify(stats), 200

@app.route('/config', methods=['GET'])
def get_config():
    """Return current configuration"""
    # Remove sensitive information if necessary
    safe_config = config.copy()
    return jsonify(safe_config), 200

if __name__ == '__main__':
    # Start worker threads
    logger.info(f"Starting {config['MAX_WORKERS']} worker threads")
    workers = []
    for _ in range(config["MAX_WORKERS"]):
        worker = threading.Thread(target=batch_processor, daemon=True)
        worker.start()
        workers.append(worker)
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_responses, daemon=True)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Start Flask server
    logger.info(f"Starting server on {config['HOST']}:{config['PORT']}")
    app.run(
        host=config["HOST"], 
        port=config["PORT"], 
        debug=config["DEBUG"],
        threaded=True
    )