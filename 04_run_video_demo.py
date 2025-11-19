import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from student_manager import get_all_students
from face_model import FaceRecognitionModel

# ==========================================
# [설정]
VIDEO_PATH = "drone_video.mp4" 

# 1. 기준 점수 대폭 완화 (0.5 -> 0.3)
# 데모에서는 일단 뭐라도 뜨게 하는 게 중요합니다.
CONFIDENCE_THRESHOLD = 0.3

# 2. 격차 기준 완화 (0.1 -> 0.05)
# 점수 차이가 미미해도 일단 1등을 보여주도록 합니다.
GAP_THRESHOLD = 0.05 

# 3. 블랙리스트 ID (계속 오인식되는 그 사람 ID)
IGNORED_IDS = [202239875] 

MODEL_PATH = 'trainer/model.pt'
# ==========================================

# --- 1. 디바이스 및 모델 설정 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found: {MODEL_PATH}")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Video file not found: {VIDEO_PATH}")
    exit()

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes = checkpoint['num_classes']
idx_to_id = checkpoint['idx_to_id']
IMG_SIZE = checkpoint.get('img_size', 100)

model = FaceRecognitionModel(num_classes).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"[INFO] Model loaded: {num_classes} classes")

# --- 2. 얼굴 검출기 및 학생 정보 ---
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

all_student_dict = get_all_students()
print(f"[INFO] Loaded {len(all_student_dict)} student(s) from database.")

# --- 3. 전처리 파이프라인 ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_face(face_roi_gray):
    try:
        face_pil = Image.fromarray(face_roi_gray) 
        face_tensor = transform(face_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            top2_prob, top2_idx = torch.topk(probabilities, 2, dim=1)
            
            conf1 = top2_prob[0][0].item()
            idx1 = top2_idx[0][0].item()
            
            if num_classes > 1:
                conf2 = top2_prob[0][1].item()
            else:
                conf2 = 0.0
            
            gap = conf1 - conf2

        # 디버깅을 위해 콘솔에 출력 (이걸 보면 왜 Unknown인지 알 수 있습니다)
        student_id = idx_to_id.get(idx1, -1)
        # print(f"[DEBUG] ID:{student_id}, Score:{conf1:.2f}, Gap:{gap:.2f}")

        # 1차 필터: 기준 점수 미달이면 탈락
        if conf1 < CONFIDENCE_THRESHOLD:
            return -1, conf1
            
        # 2차 필터: Gap이 너무 작으면 탈락
        if gap < GAP_THRESHOLD:
            return -1, conf1
        
        # 3차 필터: 블랙리스트 ID면 강제로 탈락
        if student_id in IGNORED_IDS:
            return -1, conf1

        return student_id, conf1
    
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return -1, 0.0

# --- 4. 비디오 재생 및 추론 루프 ---
cap = cv2.VideoCapture(VIDEO_PATH)
print("[INFO] Video processing started... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video ended.")
        break
    
    # 1. 흑백 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 얼굴 검출
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        
        # 예측
        student_id, confidence = predict_face(face_roi)
        confidence_percent = round(confidence * 100)
        
        display_name = "Unknown"
        color = (0, 0, 255) # Red
        
        # student_id가 -1이 아니면 (필터 통과)
        if student_id != -1:
            name = all_student_dict.get(student_id, "ID not registered")
            if name != "ID not registered":
                display_name = name
                color = (0, 255, 0) # Green
        
        # 3. 그리기
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # 화면에 점수도 같이 표시해서 디버깅 (데모 녹화할 땐 퍼센트만 나오게 수정하세요)
        label = f"{display_name} ({confidence_percent}%)"
        
        # 글자 배경 박스
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(frame, (x, y - 25), (x + tw, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), font, 0.6, (255, 255, 255), 2)

    cv2.imshow('Uhm-Tendance Demo Video', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()