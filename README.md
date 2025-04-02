# Fire Detection Microservice Documentation

## Overview

The Fire Detection Microservice is a Flask-based web service that uses a YOLOv5 model to detect fires in images. The service accepts base64-encoded images and returns detection results (bounding boxes, confidence scores, and class IDs). It's designed for efficient batch processing of images using a queue-based architecture and multiple worker threads.

## Features

- **Real-time Fire Detection**: Detect fires in images using YOLOv5 custom model
- **Efficient Batch Processing**: Process multiple images in batches for improved throughput
- **Queue-based Architecture**: Handle multiple concurrent requests with a queue system
- **GPU Acceleration**: Optional GPU support for faster inference
- **Health Monitoring**: Health check endpoint for monitoring service status
- **Configurable Parameters**: Adjust performance parameters through environment variables

## API Endpoints

### 1. Detect Fires (`POST /detect`)

Detects fires in an image.

**Request:**

```json
{
  "frame": "base64_encoded_image_string"
}
```

**Response:**

```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class_id": 0
    }
  ],
  "timestamp": 1712076422.34567
}
```

**Error Responses:**

- `400 Bad Request`: Missing or invalid image data
- `408 Request Timeout`: Processing took too long
- `500 Internal Server Error`: Error during processing
- `503 Service Overloaded`: Queue is full

### 2. Health Check (`GET /health`)

Returns the current status of the service.

**Response:**

```json
{
  "status": "fire detection service is running",
  "device": "cpu",
  "queue_usage": "0/32",
  "workers": 4,
  "batch_size": 4,
  "pending_responses": 0
}
```

### 3. Configuration (`GET /config`)

Returns the current service configuration.

**Response:**

```json
{
  "MODEL_PATH": "models/yolov5s_best.pt",
  "MAX_WORKERS": 4,
  "BATCH_SIZE": 4,
  "QUEUE_SIZE": 32,
  "PORT": 5000,
  "HOST": "0.0.0.0",
  "DEBUG": false,
  "LOG_LEVEL": "INFO",
  "CONFIDENCE_THRESHOLD": 0.25,
  "USE_GPU": "auto"
}
```

## Configuration

The service can be configured using environment variables:

| Variable               | Description                            | Default                  |
| ---------------------- | -------------------------------------- | ------------------------ |
| `MODEL_PATH`           | Path to the YOLOv5 model file          | `models/yolov5s_best.pt` |
| `MAX_WORKERS`          | Number of worker threads               | `4`                      |
| `BATCH_SIZE`           | Number of images to process in a batch | `4`                      |
| `QUEUE_SIZE`           | Maximum size of the request queue      | `32`                     |
| `PORT`                 | Server port                            | `5000`                   |
| `HOST`                 | Server host                            | `0.0.0.0`                |
| `DEBUG`                | Enable Flask debug mode                | `false`                  |
| `LOG_LEVEL`            | Logging level                          | `INFO`                   |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for detections      | `0.25`                   |
| `USE_GPU`              | GPU usage: "true", "false", or "auto"  | `auto`                   |

## Installation and Setup

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)
- YOLOv5 dependencies

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fire-detection.git
   cd fire-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the YOLOv5 model is in the specified location (default: `models/yolov5s_best.pt`).

### Running the Service

Start the service with default parameters:

```bash
python fire_detection_service.py
```

With custom configuration:

```bash
MODEL_PATH=models/custom_model.pt MAX_WORKERS=8 BATCH_SIZE=8 python fire_detection_service.py
```

## Testing

A test script (`test.py`) is provided to verify the service is working correctly:

```bash
python test.py --image test_image.jpg
```

For more testing options:

```bash
python test.py --help
```

## Architecture

The service uses a queue-based architecture with multiple worker threads:

1. **Request Handling**: The Flask server receives image requests and places them in a queue
2. **Batch Processing**: Worker threads process images in batches for efficient inference
3. **Response Management**: Results are stored in a thread-safe dictionary and returned to clients
4. **Cleanup Process**: A background thread removes old responses to prevent memory leaks

## Performance Considerations

- **Batch Size**: Larger batch sizes improve throughput but increase latency
- **Worker Threads**: More workers can handle more concurrent requests but may overload the CPU
- **Queue Size**: Larger queues allow more pending requests but use more memory
- **GPU Acceleration**: Significantly improves performance for large batches
- **Timeout**: Adjust the timeout (currently 10 seconds) based on expected processing time

## Troubleshooting

### Common Issues

1. **Service Overloaded**: If you receive a 503 error, try increasing the queue size or reducing the request rate
2. **Request Timeout**: If requests time out, try increasing the worker threads or reducing the batch size
3. **Memory Issues**: If the service uses too much memory, reduce the queue size or batch size
4. **GPU Out of Memory**: Try reducing the batch size if using GPU acceleration

### Logs

The service logs information to help with debugging:

- Check the log level (configurable via `LOG_LEVEL`)
- Look for error messages in the logs
- The health endpoint provides useful operational metrics

## Deployment

For production deployment:

1. Use a proper WSGI server like Gunicorn instead of Flask's development server
2. Consider containerizing the application with Docker
3. Set up monitoring and alerts
4. Implement appropriate security measures
5. Configure load balancing for high availability

---

This documentation provides an overview of the Fire Detection Microservice. For further assistance or to report issues, please contact the development team.
