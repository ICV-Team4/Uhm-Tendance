"""
Dummy ZMQ Image Sender
Sends a random demo image from the internet to the attendance server via ZMQ.
"""

import zmq
import requests
import cv2
import numpy as np
import time
import io
from PIL import Image

# Configuration
ZMQ_PORT = 3389  # Port that attendance server is listening on
ZMQ_HOST = "localhost"
SEND_INTERVAL = 0.1  # Send image every 0.1 seconds (10 FPS)

# Random image URLs (images with people from internet)
IMAGE_URLS = [
    "https://randomuser.me/api/portraits/men/1.jpg",  # Random user portraits - men
    "https://randomuser.me/api/portraits/women/1.jpg",  # Random user portraits - women
    "https://i.pravatar.cc/640",  # Avatar service with people
    "https://randomuser.me/api/portraits/men/32.jpg",  # Another random portrait
    "https://randomuser.me/api/portraits/women/44.jpg",  # Another random portrait
]

def download_image(url):
    """Download an image from URL and return as numpy array"""
    try:
        # Add headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=5, headers=headers)
        response.raise_for_status()
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(response.content))
        
        # Resize to 640x480 if needed
        if img.size != (640, 480):
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
        
        # Convert to OpenCV format (BGR)
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
            
        return img_bgr
    except Exception as e:
        print(f"[ERROR] Failed to download image from {url}: {e}")
        return None

def create_dummy_image():
    """Create a dummy image with a simple face-like shape if internet download fails"""
    # Create a simple colored image with a face-like shape
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (200, 180, 160)  # Skin-tone background
    
    center_x, center_y = 320, 240
    
    # Draw a simple face
    # Head (circle)
    cv2.circle(img, (center_x, center_y), 100, (220, 200, 180), -1)
    cv2.circle(img, (center_x, center_y), 100, (150, 120, 100), 2)
    
    # Eyes
    cv2.circle(img, (center_x - 30, center_y - 20), 10, (50, 50, 50), -1)
    cv2.circle(img, (center_x + 30, center_y - 20), 10, (50, 50, 50), -1)
    
    # Nose
    cv2.ellipse(img, (center_x, center_y + 10), (5, 15), 0, 0, 180, (150, 120, 100), 2)
    
    # Mouth
    cv2.ellipse(img, (center_x, center_y + 40), (20, 10), 0, 0, 180, (100, 50, 50), 2)
    
    # Add text
    cv2.putText(img, "Dummy Face", (220, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    
    return img

def main():
    print(f"[INFO] Starting dummy ZMQ image sender...")
    print(f"[INFO] Target: tcp://{ZMQ_HOST}:{ZMQ_PORT}")
    print(f"[INFO] Send interval: {SEND_INTERVAL} seconds ({1/SEND_INTERVAL} FPS)")
    
    # Setup ZMQ PUSH socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")
    print(f"[INFO] Connected to ZMQ server at tcp://{ZMQ_HOST}:{ZMQ_PORT}")
    
    # Try to download an image from internet
    print("[INFO] Attempting to download demo image from internet...")
    image = None
    for url in IMAGE_URLS:
        print(f"[INFO] Trying: {url}")
        image = download_image(url)
        if image is not None:
            print(f"[INFO] Successfully downloaded image from {url}")
            break
    
    # If download failed, create a dummy image
    if image is None:
        print("[WARNING] Failed to download image. Creating dummy image...")
        image = create_dummy_image()
    
    print("[INFO] Starting to send images... (Press Ctrl+C to stop)")
    
    frame_count = 0
    try:
        while True:
            # Encode image as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, jpeg_bytes = cv2.imencode('.jpg', image, encode_param)
            
            # Send via ZMQ
            socket.send(jpeg_bytes.tobytes())
            frame_count += 1
            
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"[INFO] Sent {frame_count} frames")
            
            time.sleep(SEND_INTERVAL)
            
    except KeyboardInterrupt:
        print(f"\n[INFO] Stopped. Total frames sent: {frame_count}")
    except Exception as e:
        print(f"[ERROR] Error occurred: {e}")
    finally:
        socket.close()
        context.term()
        print("[INFO] ZMQ connection closed.")

if __name__ == "__main__":
    main()

