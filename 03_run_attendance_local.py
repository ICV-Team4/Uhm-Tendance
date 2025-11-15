import cv2
import os
import csv
import torch
import numpy as np
from datetime import datetime
from PIL import Image
from torchvision import transforms
from student_manager import get_all_students
from face_model import FaceRecognitionModel

import base64
import json
import asyncio
import websockets 
import threading
import queue

import re
import base64
import cv2
import numpy as np

def decode_b64_image(b64_string):
    # 개행/공백 제거
    cleaned = re.sub(r'\s+', '', b64_string)

    # 패딩 복원
    padding = len(cleaned) % 4
    if padding != 0:
        cleaned += '=' * (4 - padding)

    try:
        img_bytes = base64.b64decode(cleaned)
    except Exception as e:
        print("[ERROR] Base64 decode failed:", e)
        return None

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

# --- 설정값 ---
CONFIDENCE_THRESHOLD = 0.7
OUTPUT_FOLDER = 'output'
MODEL_PATH = 'trainer/model.pt'

MOCK_SERVER_URL = "ws://127.0.0.1:8080" 
WEBSOCKET_PORT = 5001

# --- 글로벌 변수 (스레드간 통신용) ---
broadcast_queue = queue.Queue()
SUBSCRIBERS = set()

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# === 1. 웹소켓 서버 로직 ===

async def handle_subscriber(websocket, path):
    print(f"[WS Server {WEBSOCKET_PORT}] Client connected: {websocket.remote_address}")
    SUBSCRIBERS.add(websocket)
    try:
        await websocket.wait_closed()
    except websockets.ConnectionClosed:
        print(f"[WS Server {WEBSOCKET_PORT}] Client disconnected: {websocket.remote_address}")
    finally:
        SUBSCRIBERS.remove(websocket)

async def broadcast_messages():
    while True:
        try:
            message = broadcast_queue.get(timeout=0.1)
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        
        if SUBSCRIBERS:
            await asyncio.wait([
                user.send(message) for user in SUBSCRIBERS
            ])

async def start_websocket_server():
    server = await websockets.serve(
        handle_subscriber,
        "0.0.0.0",
        WEBSOCKET_PORT
    )
    print(f"[WS Server {WEBSOCKET_PORT}] Hwa's Server started at ws://0.0.0.0:{WEBSOCKET_PORT}")
    await broadcast_messages()

def run_server_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_websocket_server())
    loop.run_forever()

# === 2. 이미지 처리 및 메인 로직 (메인 스레드) ===

async def main_async():
    # --- 2-1. 웹소켓 서버 스레드 시작 ---
    server_thread = threading.Thread(target=run_server_loop, daemon=True)
    server_thread.start()
    print(f"[INFO] Hwa's WebSocket server thread (port {WEBSOCKET_PORT}) started.")

    # --- 2-2. 모델 및 학생 정보 로드 ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {DEVICE}")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        exit()
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    num_classes = checkpoint['num_classes']
    idx_to_id = checkpoint['idx_to_id']
    IMG_SIZE = checkpoint.get('img_size', 100)
    model = FaceRecognitionModel(num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[INFO] Model loaded: {num_classes} classes")

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    if faceCascade.empty():
        print("[ERROR] Cascade classifier not loaded.")
        exit()

    all_student_dict = get_all_students()
    print(f"[INFO] Loaded {len(all_student_dict)} student(s) from database.")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    def predict_face(face_roi):
        try:
            face_img = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_pil = Image.fromarray(face_img)
            face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()
            
            student_id = idx_to_id.get(predicted_idx, -1)
            return student_id, confidence
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            return -1, 0.0

    # --- 2-3. 메인 루프 ---
    attendance_log = {}
    frame_id_counter = 0
    print("[INFO] Starting real-time attendance system... (Press 'q' in CV window to quit)")
    
    try:
        # Mock 서버 (8080)에 접속
        async with websockets.connect(MOCK_SERVER_URL) as websocket:
            print(f"[WS Client] Connected to Mock Server at {MOCK_SERVER_URL}")
            
            # Mock 서버에서 메시지(JSON)를 실시간으로 받음
            async for message in websocket:
                
                # 1. Mock 서버의 JSON 데이터 파싱
                try:
                    data = json.loads(message)

                    if 'raw_frame' not in data or 'data' not in data['raw_frame']:
                        print(f"[WS Client] Unknown JSON format received: {data.keys()}")
                        continue
                        
                    raw_frame_base64 = data['raw_frame']['data']
                    
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"[WS Client] Error processing message: {e}")
                    continue

                # 2. JPEG 바이트 -> OpenCV 프레임으로 디코딩
                frame = decode_b64_image(raw_frame_base64)

                if frame is None:
                    print("[ERROR] Failed to decode frame from mock server.")
                    continue
                
                # 원본 프레임 (Base64 인코딩용)
                _, raw_frame_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                raw_frame_base64_out = base64.b64encode(raw_frame_jpeg).decode('utf-8')

                # 3. 얼굴 인식 로직 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                boxes = []
                scores = []
                names_list = []

                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    student_id, confidence = predict_face(face_roi)
                    confidence_percent = round(confidence * 100)
                    
                    display_name = "Unknown"
                    color = (0, 0, 255) 
                    
                    if confidence >= CONFIDENCE_THRESHOLD and student_id != -1:
                        name = all_student_dict.get(student_id, "ID not registered")
                        if name != "ID not registered":
                            display_name = name
                            color = (0, 255, 0)
                            if student_id not in attendance_log:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                attendance_log[student_id] = {'name': name, 'timestamp': timestamp}
                                print(f"[ATTENDANCE] {name} (ID: {student_id}) checked in at {timestamp}")

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    cv2.putText(frame, f"{display_name} ({confidence_percent}%)", 
                               (x+5, y-5), font, 0.6, color, 2)
                    
                    boxes.append([int(x), int(y), int(x+w), int(y+h)])
                    scores.append(float(confidence))
                    names_list.append(display_name)
                
                _, annotated_frame_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                annotated_frame_base64 = base64.b64encode(annotated_frame_jpeg).decode('utf-8')

                # 6. 실시간 'frame_bundle' JSON 생성 
                frame_id_counter += 1
                frame_bundle = {
                    "version": 1,
                    "type": "frame_bundle",
                    "frame_id": frame_id_counter,
                    "timestamp": datetime.now().isoformat(),
                    "raw_frame": { "format": "jpeg", "data": raw_frame_base64_out },
                    "annotated_frame": { "format": "jpeg", "data": annotated_frame_base64 },
                    "boxes": boxes,
                    "scores": scores,
                    "names": names_list
                }
                
                # 7. 5001번 서버로 방송 
                broadcast_queue.put(json.dumps(frame_bundle))

                # 8. 로컬 창에 보여주기
                cv2.imshow('Real-time Attendance (TEST Mode from 8080)', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except websockets.exceptions.ConnectionClosedError:
        print(f"[WS Client] Mock server (8080) connection closed.")
    except ConnectionRefusedError:
        print(f"[ERROR] Connection to Mock Server (8080) refused. Is it running?")
    except Exception as e:
        print(f"[ERROR] An error occurred in the main loop: {e}")
    finally:
        # --- 2-5. 종료 처리 ---
        print("[INFO] Stopping attendance system...")
        cv2.destroyAllWindows()

        today_date = datetime.now().strftime("%Y-%m-%d")
        output_csv_path = os.path.join(OUTPUT_FOLDER, f'attendance_{today_date}.csv')
        with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['학번', '이름', '출석 시간'])
            for student_id in sorted(attendance_log.keys()):
                student_info = attendance_log[student_id]
                writer.writerow([student_id, student_info['name'], student_info['timestamp']])
        print(f"[INFO] Report saved to: {output_csv_path}")

        all_student_ids = set(all_student_dict.keys())
        attended_student_ids = set(attendance_log.keys())
        absent_student_ids = sorted(list(all_student_ids - attended_student_ids))

        attended_list_for_json = [{'id': sid, 'name': info['name'], 'timestamp': info['timestamp']} for sid, info in sorted(attendance_log.items())]
        absent_list_for_json = [{'id': sid, 'name': all_student_dict.get(sid, "Unknown")} for sid in absent_student_ids]

        report_data = {
            'type': 'attendance_report',
            'timestamp': datetime.now().isoformat(),
            'attended': attended_list_for_json,
            'absent': absent_list_for_json
        }
        report_json = json.dumps(report_data, ensure_ascii=False, indent=2)
        broadcast_queue.put(report_json)
        
        print(f"[INFO] Final report sent to Hwa's server ({WEBSOCKET_PORT}).")
        print("[INFO] Attendance check process finished.")
        
        import time
        time.sleep(1) 


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user.")