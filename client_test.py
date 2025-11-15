import cv2
import zmq
import numpy as np
import time

# --- "cam.uhmcv.kro.kr" -> "127.0.0.1" (로컬 테스트용) ---
SERVER_IP = "127.0.0.1" 
PORT = 5555

print("[INFO] ZMQ Context 생성 중...")
context = zmq.Context()
socket = context.socket(zmq.PUSH)

print(f"[INFO] Hwa님의 로컬 서버(ZMQ PULL)에 연결 시도: tcp://{SERVER_IP}:{PORT}")
socket.connect(f"tcp://{SERVER_IP}:{PORT}")
print(f"[INFO] 연결 성공! tcp://{SERVER_IP}:{PORT}")

print("[INFO] MacBook 웹캠을 켜는 중...")
cap = cv2.VideoCapture(0) # 0번 카메라 (MacBook 기본 웹캠)

if not cap.isOpened():
    print("[ERROR] MacBook 웹캠을 열 수 없습니다!")
    exit()

print("[INFO] 웹캠 캡처 및 ZMQ 전송 시작... (q: 종료)")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] 웹캠에서 프레임을 읽을 수 없습니다.")
            continue

        # Hwa님의 서버(server.py)가 받을 수 있도록 JPEG로 인코딩
        ret_encode, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret_encode:
            print("[ERROR] JPEG 인코딩 실패")
            continue

        # ZMQ 5000번 포트로 JPEG 바이트 전송
        socket.send(buffer.tobytes())

        # 웹캠 원본 보기
        cv2.imshow("ZMQ Client - My MacBook Webcam (q: 종료)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] 'q' 키로 클라이언트를 종료합니다.")
            break

        time.sleep(0.01) # (CPU 사용량 조절)

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C로 클라이언트를 종료합니다.")

finally:
    print("[INFO] 리소스 정리 중...")
    cap.release()
    socket.close()
    context.term()
    cv2.destroyAllWindows()
    print("[INFO] ZMQ 클라이언트 테스트 종료.")