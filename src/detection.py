# detection.py
import cv2
import datetime
import numpy as np
from .config import model, CONFIDENCE_THRESHOLD

def detect_fire(frame):
    """
    Detect fire in a given frame.
    Returns annotated frame and a list of detections.
    """
    try:
        # Convert frame to RGB for the model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError("Error converting image to RGB") from e

    results = model(frame_rgb)
    predictions = results.pandas().xyxy[0]
    detections = []
    annotated_frame = frame.copy()

    for _, prediction in predictions.iterrows():
        confidence = prediction.get('confidence', 0)
        if confidence >= CONFIDENCE_THRESHOLD:
            try:
                x1, y1, x2, y2 = map(int, [prediction['xmin'], prediction['ymin'], prediction['xmax'], prediction['ymax']])
            except Exception as e:
                continue  # Skip this prediction if conversion fails

            label = prediction.get('name', 'unknown')
            frame_area = annotated_frame.shape[0] * annotated_frame.shape[1]
            detection_area = (x2 - x1) * (y2 - y1)
            area_percentage = (detection_area / frame_area) * 100

            # Choose color based on confidence
            box_color = (0, 0, 255) if confidence > 0.8 else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            text = f"{label}: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), 
                          (x1 + text_size[0] + 10, y1), box_color, -1)
            cv2.putText(annotated_frame, text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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

    # Add timestamp overlay
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, timestamp, 
                (annotated_frame.shape[1] - 200, annotated_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return {
        "annotated_frame": annotated_frame,
        "detections": detections
    }
