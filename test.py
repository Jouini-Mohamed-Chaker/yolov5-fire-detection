import requests
import base64
import cv2
import numpy as np
import os
import argparse
import time
import sys
from datetime import datetime

def test_fire_detection(image_path, server_url="http://localhost:5000", save_output=True, output_dir="test_results"):
    """
    Test the fire detection service by sending an image and visualizing the results.
    
    Args:
        image_path (str): Path to the test image
        server_url (str): URL of the fire detection server
        save_output (bool): Whether to save the output image
        output_dir (str): Directory to save output images
    
    Returns:
        tuple: (processed_image, detection_count, response_time)
    """
    try:
        start_time = time.time()
        
        # Create output directory if it doesn't exist
        if save_output and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Load and encode image
        print(f"Loading image from {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Get image dimensions for later use
        height, width = image.shape[:2]
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        # Send request to the server
        print(f"Sending request to {server_url}/detect")
        response = requests.post(
            f"{server_url}/detect",
            json={"frame": encoded_image}
        )
        
        response_time = time.time() - start_time
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            detections = result["detections"]
            print(f"Found {len(detections)} objects in {response_time:.3f} seconds")
            
            # Create a copy of the image for visualization
            output_image = image.copy()
            
            # Define colors for bounding boxes (BGR format)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            
            # Draw bounding boxes and labels
            for i, detection in enumerate(detections):
                bbox = detection["bbox"]
                confidence = detection["confidence"]
                class_id = detection["class_id"]
                
                # Convert bbox coordinates to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get color for this detection
                color = colors[i % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text
                label = f"Fire: {confidence:.2f}"
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    output_image, 
                    (x1, y1 - text_height - 5), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    output_image, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2
                )
            
            # Save the output image if requested
            if save_output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = os.path.join(output_dir, f"detection_{timestamp}.jpg")
                cv2.imwrite(output_filename, output_image)
                print(f"Results saved to {output_filename}")
            
            # Show the results - using a different approach for clean exit
            print("Displaying results. Press any key to continue or Ctrl+C to exit.")
            cv2.imshow("Fire Detection Results", output_image)
            
            # Use a timed waitKey with small intervals to allow for interruption
            key = -1
            while key == -1:
                try:
                    # Wait for a short time to allow keyboard interrupt
                    key = cv2.waitKey(100)
                except KeyboardInterrupt:
                    break
            
            cv2.destroyAllWindows()
            
            return output_image, len(detections), response_time
        else:
            print(f"Error: {response.status_code}")
            if response.headers.get('content-type') == 'application/json':
                print(response.json().get('error', 'Unknown error'))
            else:
                print(f"Response: {response.text}")
            return None, 0, response_time
        
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting...")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        cv2.destroyAllWindows()
        return None, 0, time.time() - start_time

def check_server_health(server_url="http://localhost:5000"):
    """Check the health of the fire detection server"""
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            print("Server health check:")
            for key, value in response.json().items():
                print(f"  {key}: {value}")
            return True
        else:
            print(f"Health check failed with status code {response.status_code}")
            return False
    except KeyboardInterrupt:
        print("\nHealth check interrupted. Exiting...")
        sys.exit(0)
    except requests.RequestException as e:
        print(f"Error connecting to server: {e}")
        return False

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Test the fire detection service")
        parser.add_argument("--image", default="test_image.jpg", help="Path to the test image")
        parser.add_argument("--url", default="http://localhost:5000", help="URL of the fire detection server")
        parser.add_argument("--health", action="store_true", help="Check server health")
        parser.add_argument("--output-dir", default="test_results", help="Directory to save output images")
        args = parser.parse_args()
        
        if args.health:
            check_server_health(args.url)
        
        test_fire_detection(args.image, args.url, output_dir=args.output_dir)
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
        cv2.destroyAllWindows()
        sys.exit(0)