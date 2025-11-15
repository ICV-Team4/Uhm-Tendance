import asyncio
import websockets
import base64
import cv2
import json
import time
import os
from datetime import datetime

# --- 설정 ---
PORT = 8080
IMAGE_PATH = "test.jpg" 

# 이미지를 읽고 Base64로 인코딩하는 함수
def get_image_base64(path):
    if not os.path.exists(path):
        print(f"[ERROR] Mock Server: 이미지 파일을 찾을 수 없습니다! 경로: {path}")
        print("[INFO] Hwa님의 dataset 폴더에 있는 실제 이미지 경로로 수정해주세요.")
        return None, None

    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Mock Server: OpenCV가 이미지를 읽지 못했습니다. 경로: {path}")
        return None, None

    # 이미지를 JPEG로 인코딩
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    
    # Base64로 인코딩 (Hwa님의 sample-data.json과 동일한 "순수" 텍스트)
    b64_data = base64.b64encode(buffer).decode('utf-8')
    
    # JSON 페이로드 생성
    frame_bundle = {
        "version": 1,
        "type": "frame_bundle",
        "frame_id": int(time.time()),
        "timestamp": datetime.now().isoformat(),
        "raw_frame": {
            "format": "jpeg",
            "data": b64_data # "순수" Base64 데이터
        },
        "annotated_frame": {
            "format": "jpeg",
            "data": b64_data # (테스트용이라 원본과 동일하게 보냄)
        },
        "boxes": [],
        "scores": [],
        "names": []
    }
    return json.dumps(frame_bundle), img.shape

# 웹소켓 서버 핸들러
async def handler(websocket):
    print(f"[Mock Server {PORT}] 클라이언트(AI서버)가 접속했습니다.")
    
    # 이미지 읽기 시도
    json_message, shape = get_image_base64(IMAGE_PATH)
    
    if json_message is None:
        print(f"[Mock Server {PORT}] 이미지 준비 실패. 서버를 종료합니다.")
        await websocket.close()
        return

    print(f"[Mock Server {PORT}] 이미지({shape})를 0.5초마다 방송합니다...")

    try:
        while True:
            await websocket.send(json_message)
            await asyncio.sleep(0.5) # 0.5초 대기
    except websockets.exceptions.ConnectionClosed:
        print(f"[Mock Server {PORT}] 클라이언트 접속이 끊겼습니다.")

# 서버 시작
async def start_server():
    async with websockets.serve(handler, "127.0.0.1", PORT):
        print(f"[Mock Server {PORT}] Python Mock 서버가 ws://127.0.0.1:{PORT} 에서 실행 중입니다.")
        await asyncio.Future()  # 영원히 실행

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print(f"\n[Mock Server {PORT}] 서버를 종료합니다.")