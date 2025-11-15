import cv2
import os
import json
import numpy as np
import base64
import re

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

# --- 메인 테스트 로직 ---
if not os.path.exists('sample-data.json'):
    print("[ERROR] 'sample-data.json' 파일을 찾을 수 없습니다!")
    exit()

print("[INFO] 'sample-data.json' 파일을 읽는 중...")
with open('sample-data.json', 'r') as f:
    data = json.load(f)

# JSON에서 첫 번째 이미지의 Base64 데이터 추출
try:
    b64_string = data[0]['raw_frame']['data']
except Exception as e:
    print(f"[ERROR] JSON 구조가 잘못되었습니다: {e}")
    exit()

print("[INFO] Base64 디코딩 및 이미지 변환 시도...")

frame = decode_b64_image(b64_string)

if frame is not None:
    print("\n[SUCCESS] 이미지 디코딩 성공!")
    print(f"[INFO] 이미지 크기: {frame.shape}")
    
    cv2.imshow("Test Image (q: 종료)", frame)
    print("[INFO] 이미지 창이 떴습니다. 'q' 키를 누르면 종료됩니다.")
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
else:
    print("\n[FAILURE] 이미지 디코딩 실패. 'frame'이 None입니다.")
    print("[INFO] 'sample-data.json'의 Base64 데이터가 손상되었거나, OpenCV가 읽을 수 없는 형식입니다.")

print("[INFO] 테스트 완료.")