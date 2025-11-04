import cv2
import os
from student_manager import add_student

# 얼굴 데이터 저장 경로
output_folder = 'dataset'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Haar Cascade 분류기 로드
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 사용자 정보 입력
face_id = input('\n Enter user id (must be a number) ==>  ')
name = input(' Enter user name ==>  ')

# 학생 ID별로 폴더 생성
student_folder_path = os.path.join(output_folder, face_id)
if not os.path.exists(student_folder_path):
    os.makedirs(student_folder_path)
    print(f"[INFO] Created directory for {name}: {student_folder_path}")

# 학생 정보 저장
add_student(face_id, name)
print(f"\n [INFO] Student {name} (ID: {face_id}) registered.")
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        
        # 얼굴 이미지를 학생 ID별 폴더에 저장 (파일 이름을 1.jpg, 2.jpg...로 저장)
        file_path = os.path.join(student_folder_path, f"{face_id}_{count}.jpg")
        cv2.imwrite(file_path, gray[y:y+h, x:x+w])

        cv2.imshow('image', frame)

    # 'q' 키를 누르거나 30장 촬영 시 종료
    if cv2.waitKey(100) & 0xFF == ord('q') or count >= 30:
        break

print(f"\n [INFO] Face capture finished. Saved {count} images to {student_folder_path}")
cap.release()
cv2.destroyAllWindows()