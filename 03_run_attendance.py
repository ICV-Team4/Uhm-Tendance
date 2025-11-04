import cv2
import os
import csv
import torch
import numpy as np
from datetime import datetime
from PIL import Image
from torchvision import transforms
from student_manager import get_all_students, get_student_name
from face_model import FaceRecognitionModel

CONFIDENCE_THRESHOLD = 0.7  # 신뢰도 임계값 (0~1)
OUTPUT_FOLDER = 'output'
MODEL_PATH = 'trainer/model.pt'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 디바이스 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# 모델 로드
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found: {MODEL_PATH}")
    print("[INFO] Please run 02_train_model.py first to train the model.")
    exit()

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes = checkpoint['num_classes']
idx_to_id = checkpoint['idx_to_id']
id_to_idx = checkpoint.get('id_to_idx', {v: k for k, v in idx_to_id.items()})
IMG_SIZE = checkpoint.get('img_size', 100)

model = FaceRecognitionModel(num_classes).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"[INFO] Model loaded: {num_classes} classes")

# 얼굴 검출기
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

if faceCascade.empty():
    print("Error: Cascade classifier not loaded. Make sure 'haarcascade_frontalface_default.xml' is in the same folder.")
    exit()

# 학생 정보 동적 로드
names = get_all_students()
if not names:
    print("[WARNING] No students registered. Please run 01_collect_data.py first.")
else:
    print(f"[INFO] Loaded {len(names)} student(s) from database.")

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_face(face_roi):
    """얼굴 영역에서 학생 ID 예측"""
    try:
        # 이미지 전처리
        face_img = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_pil = Image.fromarray(face_img)
        face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
        
        # 예측
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_idx = predicted_idx.item()
        
        # 인덱스를 실제 학생 ID로 변환
        student_id = idx_to_id.get(predicted_idx, -1)
        
        return student_id, confidence
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return -1, 0.0

cap = cv2.VideoCapture(0)
attendance_log = {}

print("[INFO] Starting real-time attendance system... (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # 얼굴 영역 추출
        face_roi = gray[y:y+h, x:x+w]
        
        # 예측
        student_id, confidence = predict_face(face_roi)
        confidence_percent = round(confidence * 100)
        
        if confidence >= CONFIDENCE_THRESHOLD and student_id != -1:
            name = get_student_name(student_id)
            
            if name != "ID not registered" and student_id not in attendance_log:
                timestamp = datetime.now().strftime("%H:%M:%S")
                attendance_log[student_id] = {'name': name, 'timestamp': timestamp}
                print(f"[ATTENDANCE] {name} (ID: {student_id}) checked in at {timestamp}")
            
            display_name = name
            color = (0, 255, 0)  # 초록색
        else:
            display_name = "Unknown"
            color = (0, 0, 255)  # 빨간색
        
        # 이름과 신뢰도 표시
        cv2.putText(frame, f"{display_name} ({confidence_percent}%)", 
                   (x+5, y-5), font, 0.6, color, 2)
    
    cv2.imshow('Real-time Attendance', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

today_date = datetime.now().strftime("%Y-%m-%d")
output_csv_path = os.path.join(OUTPUT_FOLDER, f'attendance_{today_date}.csv')

with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['학번', '이름', '출석 시간'])
    
    # 출석 로그를 학번 순으로 정렬하여 저장
    for student_id in sorted(attendance_log.keys()):
        student_info = attendance_log[student_id]
        writer.writerow([student_id, student_info['name'], student_info['timestamp']])

print(f"[INFO] Report saved to: {output_csv_path}")

cap.release()
cv2.destroyAllWindows()
