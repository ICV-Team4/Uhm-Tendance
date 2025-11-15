UHM-TENDANCE: Tello 드론 연동 얼굴 인식 출석 시스템

PyTorch 기반 얼굴 인식을 통해 Tello 드론의 ZMQ 영상 스트림을 실시간으로 분석하고, WebSocket을 통해 출석 결과를 방송(broadcast)하는 서버입니다.

주요 기능

실시간 ZMQ 입력: Tello 드론(클라이언트)이 전송하는 JPEG 영상 스트림을 ZMQ PULL 소켓으로 수신합니다.

실시간 얼굴 인식: 수신된 매 프레임마다 PyTorch 모델을 사용해 얼굴을 인식하고 students.json의 정보와 대조합니다.

실시간 WebSocket 방송: 인식 결과(Base64 이미지, 좌표, 이름)를 frame_bundle JSON 형식으로 WebSocket 서버를 통해 모든 클라이언트(웹 대시보드)에 방송합니다.

최종 리포트: 프로그램 종료 시, output/ 폴더에 CSV 리포트를 저장하고, attendance_report JSON을 WebSocket으로 마지막 방송합니다.

설치 방법

1. (최초 1회) 가상환경 및 패키지 설치

# (icv 가상환경이 이미 있다고 가정)
# Tello 드론 및 ZMQ/WebSocket에 필요한 패키지 설치
pip install -r requirements.txt



requirements.txt에 아래 패키지들이 포함되어 있는지 확인하세요:

opencv-python
torch
torchvision
Pillow
numpy
websockets
pyzmq
djitellopy
ultralytics



파일 구조 (주요 파일)

Uhm-Tendance/
├── 01_collect_data.py           # (데이터 수집용) 학생 얼굴 데이터 수집
├── 02_train_model.py            # (데이터 수집용) PyTorch 모델 학습
├── 03_run_attendance_server.py  # (★★실제 AI 서버★★) ZMQ 입력 -> WS 출력
├── zmq_client_test_LOCAL.py     # (테스트용) "가짜 텔로" (MacBook 웹캠 ZMQ PUSH)
├── mock_server_PYTHON.py        # (테스트용) "가짜 텔로" (Python WS PUSH)
├── mock_server.js               # (테스트용) "가짜 텔로" (Node.js WS PUSH)
├── face_model.py                # PyTorch 모델 정의
├── student_manager.py           # 학생 정보 관리
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── dataset/                     # 학생 얼굴 이미지
├── trainer/
│   └── model.pt               # 최종 학습된 모델
├── output/                      # 출석 기록 CSV
└── students.json                # 학생 정보 (학번: 이름)



(선택) 1. 모델 학습 방법

trainer/model.pt 파일이 없는 경우, 먼저 모델을 학습해야 합니다.

# 1. 학생 얼굴 데이터 수집 (웹캠 필요)
python 01_collect_data.py

# 2. 모델 학습
python 02_train_model.py



(중요) 2. AI 서버 실행 방법

AI 서버(03_run_attendance_server.py)는 ZMQ로 영상을 받고 WebSocket으로 결과를 방송합니다.

A. 로컬 테스트 (Local Test)

macOS의 경우 5000번 포트가 AirPlay와 충돌할 수 있으므로, 로컬 테스트 시 **5555**번 포트를 사용하도록 기본 설정되어 있습니다.

터미널 2개가 필요합니다.

$$터미널 1: AI 서버 실행$$

# (ZMQ_PORT 환경변수 설정 안 함 -> 5555번 포트로 자동 실행됨)
/opt/anaconda3/envs/icv/bin/python 03_run_attendance_server.py



[ZMQ] Setting up ZMQ PULL socket at tcp://*:5555
[WS Server 5556] WebSocket Server started at ws://0.0.0.0:5556
(두 메시지가 뜨고 대기하는지 확인)

$$터미널 2: ZMQ 테스트 클라이언트 (웹캠) 실행$$

# 로컬 MacBook 웹캠 영상을 ZMQ 5555번 포트로 전송
/opt/anaconda3/envs/icv/bin/python zmq_client_test_LOCAL.py



[INFO] 연결 성공! tcp://127.0.0.1:5555
(웹캠이 켜지고, 터미널 1에서 얼굴 인식 로그가 뜨기 시작함)

$$결과 확인$$

웹 대시보드에서 ws://localhost:5556에 접속하면 실시간 결과를 볼 수 있습니다.

B. 실제 서버 배포 (Production Server)

배포 서버(Linux 등)는 5000번 포트 충돌이 없으므로, 텔로 드론 ZMQ 클라이언트(5000) 포트에 맞춰 실행합니다.

$$서버 터미널: AI 서버 실행$$

# 1. (중요) ZMQ_PORT 환경변수를 5000으로 설정
export ZMQ_PORT=5000

# 2. AI 서버 실행
python 03_run_attendance_server.py



[ZMQ] Setting up ZMQ PULL socket at tcp://*:5000
[WS Server 5556] WebSocket Server started at ws://0.0.0.0:5556
(두 메시지가 뜨고 대기하는지 확인)

$$결과 확인$$

텔로 드론 클라이언트를 cam.uhmcv.kro.kr:5000으로 연결하면, 서버가 자동으로 인식을 시작합니다.

웹 대시보드에서 ws://cam.uhmcv.kro.kr:5556에 접속하면 실시간 결과를 볼 수 있습니다.

포트 정리

| 서비스 | 포트 | 설명 |
| ZMQ (Tello -> AI서버) | 5000 | (실서버용) 텔로 드론 원본 영상 |
| WebSocket (AI서버 -> 웹) | 5556 | (최종) AI 서버 방송 포트 |
| ZMQ (Local Test) | 5555 | (macOS 로컬 테스트용 포트) |