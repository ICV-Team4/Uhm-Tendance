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

# --- 새롭게 추가된 라이브러리 ---
import zmq        # 1. ZMQ로 Tello 이미지 수신
import base64     # 2. 이미지를 Base64로 인코딩
import json       # 3. JSON 메시지 생성
import asyncio    # 4. 비동기 웹소켓 서버
import websockets # 5. 웹소켓 서버
import threading  # 6. OpenCV(Main)와 웹소켓(Server)을 분리
import queue      # 7. 메인 스레드 -> 서버 스레드로 데이터 전송

import re
import base64
import cv2
import numpy as np

def decode_b64_image(b64_string):
    cleaned = re.sub(r'\s+', '', b64_string)
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

ZMQ_PORT = int(os.getenv("ZMQ_PORT", 5555)) 

WEBSOCKET_PORT = 5556   # 인식 결과를 방송할 포트

# --- 글로벌 변수 (스레드간 통신용) ---
broadcast_queue = queue.Queue() 
SUBSCRIBERS = set()             

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# === 1. 웹소켓 서버 로직 (별도 스레드에서 실행) ===
async def handle_subscriber(websocket, path=None):
    """
    새 클라이언트가 접속하면 SUBSCRIBERS 세트에 추가하고,
    접속이 끊기면 제거합니다.
    """
    print(f"[WS Server {WEBSOCKET_PORT}] Client connected: {websocket.remote_address}")
    SUBSCRIBERS.add(websocket)
    try:
        await websocket.wait_closed()
    except websockets.ConnectionClosed:
        print(f"[WS Server {WEBSOCKET_PORT}] Client disconnected: {websocket.remote_address}")
    finally:
        SUBSCRIBERS.remove(websocket)

async def broadcast_messages():
    """
    broadcast_queue에 메시지가 들어올 때까지 기다렸다가
    모든 접속자에게 메시지를 전송합니다.
    """
    while True:
        try:
            message = broadcast_queue.get(timeout=0.1)
        except queue.Empty:
            await asyncio.sleep(0.01) 
            continue
        
        if SUBSCRIBERS:
            await asyncio.gather(
                *[user.send(message) for user in SUBSCRIBERS],
                return_exceptions=True
            )

async def start_websocket_server():
    """비동기 웹소켓 서버 시작"""
    server = await websockets.serve(
        handle_subscriber,
        "0.0.0.0",  
        WEBSOCKET_PORT
    )
    print(f"[WS Server {WEBSOCKET_PORT}] WebSocket Server started at ws://0.0.0.0:{WEBSOCKET_PORT}")
    await broadcast_messages() 

def run_server_loop():
    """
    새로운 이벤트 루프를 생성하고 그 안에서 웹소켓 서버를 실행
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_websocket_server())
    loop.run_forever()

# === 2. 이미지 처리 및 메인 로직 (메인 스레드) ===

def main():
    # --- 2-1. 웹소켓 서버 스레드 시작 ---
    server_thread = threading.Thread(target=run_server_loop, daemon=True)
    server_thread.start()
    print(f"[INFO] WebSocket server thread (port {WEBSOCKET_PORT}) started.")

    # --- 2-2. ZMQ PULL 소켓 설정 ---
    print(f"[ZMQ] Setting up ZMQ PULL socket at tcp://*:{ZMQ_PORT}")
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PULL)
    zmq_socket.bind(f"tcp://*:{ZMQ_PORT}")
    print("[ZMQ] Ready to receive frames from Tello drone...")

    # --- 2-3. 모델 및 학생 정보 로드 ---
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
    if not all_student_dict:
        print("[WARNING] No students registered.")
    else:
        print(f"[INFO] Loaded {len(all_student_dict)} student(s) from database.")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    def predict_face(face_roi):
        """얼굴 영역에서 학생 ID 예측"""
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

    # --- 2-4. 메인 루프 (ZMQ + OpenCV) ---
    attendance_log = {}
    frame_id_counter = 0
    print("[INFO] Starting real-time attendance system... (Press 'q' in CV window to quit)")

    while True:
        # 1. ZMQ로 Tello 이미지 수신 (JPEG 바이트)
        try:
            jpeg_bytes = zmq_socket.recv()
        except zmq.ZMQError as e:
            print(f"[ZMQ] Error receiving frame: {e}")
            continue

        # 2. JPEG 바이트 -> OpenCV 프레임으로 디코딩
        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("[ERROR] Failed to decode frame.")
            continue
        
        # 원본 프레임 (Base64 인코딩용)
        _, raw_frame_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        raw_frame_base64 = base64.b64encode(raw_frame_jpeg).decode('utf-8')

        # 3. 얼굴 인식 로직 (기존 코드 활용)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # JSON용 데이터 리스트
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

            # 4. 인식 결과 프레임에 그리기
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{display_name} ({confidence_percent}%)", 
                       (x+5, y-5), font, 0.6, color, 2)
            
            boxes.append([int(x), int(y), int(x+w), int(y+h)]) 
            scores.append(float(confidence))
            names_list.append(display_name)
        
        # 5. 인식 결과가 그려진 프레임 (Base64 인코딩용)
        _, annotated_frame_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        annotated_frame_base64 = base64.b64encode(annotated_frame_jpeg).decode('utf-8')

        # 6. 실시간 'frame_bundle' JSON 생성 (다이어그램 스키마)
        frame_id_counter += 1
        frame_bundle = {
            "version": 1,
            "type": "frame_bundle",
            "frame_id": frame_id_counter,
            "timestamp": datetime.now().isoformat(),
            "raw_frame": {
                "format": "jpeg",
                "data": raw_frame_base64
            },
            "annotated_frame": {
                "format": "jpeg",
                "data": annotated_frame_base64
            },
            "boxes": boxes,
            "scores": scores,
            "names": names_list
        }
        
        # 7. JSON을 큐에 넣어 웹소켓 서버 스레드로 전송
        broadcast_queue.put(json.dumps(frame_bundle))

    # --- 2-5. 종료 처리 ---
    print("[INFO] Stopping attendance system...")
    zmq_socket.close()
    context.term()
    cv2.destroyAllWindows()

    # --- CSV 저장 ---
    today_date = datetime.now().strftime("%Y-%m-%d")
    output_csv_path = os.path.join(OUTPUT_FOLDER, f'attendance_{today_date}.csv')
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['학번', '이름', '출석 시간'])
        for student_id in sorted(attendance_log.keys()):
            student_info = attendance_log[student_id]
            writer.writerow([student_id, student_info['name'], student_info['timestamp']])
    print(f"[INFO] Report saved to: {output_csv_path}")

    # --- 최종 리포트 생성 ---
    all_student_ids = set(all_student_dict.keys())
    attended_student_ids = set(attendance_log.keys())
    absent_student_ids = sorted(list(all_student_ids - attended_student_ids))

    attended_list_for_json = []
    for student_id, info in sorted(attendance_log.items()):
        attended_list_for_json.append({
            'id': student_id,
            'name': info['name'],
            'timestamp': info['timestamp']
        })

    absent_list_for_json = []
    for student_id in absent_student_ids:
        absent_list_for_json.append({
            'id': student_id,
            'name': all_student_dict.get(student_id, "Unknown")
        })

    # --- 최종 리포트 JSON을 큐에 넣어 방송 ---
    report_data = {
        'type': 'attendance_report', # 리포트 타입
        'timestamp': datetime.now().isoformat(),
        'attended': attended_list_for_json,
        'absent': absent_list_for_json
    }
    report_json = json.dumps(report_data, ensure_ascii=False, indent=2)
    broadcast_queue.put(report_json)
    
    print("[INFO] Final report sent to all subscribers.")
    print("[INFO] Attendance check process finished.")
    
    # 서버 스레드가 메시지를 보낼 수 있도록 잠시 대기
    import time
    time.sleep(1)

if __name__ == "__main__":
    main()