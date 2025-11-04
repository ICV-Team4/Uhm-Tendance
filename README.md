# 얼굴 인식 출석 시스템

PyTorch 기반 얼굴 인식을 통한 실시간 출석 체크 시스템입니다.

## 설치 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install opencv-python torch torchvision Pillow numpy
```

## 사용 방법

### 단계 1: 학생 얼굴 데이터 수집

```bash
python 01_collect_data.py
```

- 실행 시 학번과 이름을 입력하세요
- 웹캠 앞에서 얼굴을 보여주면 자동으로 30장 촬영됩니다
- 학생 정보는 `students.json`에 자동 저장됩니다
- 'q' 키를 누르면 중간에 종료할 수 있습니다

### 단계 2: 모델 학습

```bash
python 02_train_model.py
```

- `dataset` 폴더의 이미지들을 학습합니다
- 학습된 모델은 `trainer/model.pt`에 저장됩니다
- 학습 진행 상황(손실, 정확도)이 표시됩니다
- GPU가 있으면 자동으로 사용합니다

### 단계 3: 출석 체크 실행

```bash
python 03_run_attendance.py
```

- 웹캠으로 실시간 얼굴 인식을 시작합니다
- 인식된 학생은 자동으로 출석 처리됩니다
- 출석 정보는 `output/attendance_YYYY-MM-DD.csv`에 저장됩니다
- 'q' 키를 누르면 종료하고 CSV 파일을 저장합니다

## 파일 구조

```
termproject/
├── 01_collect_data.py          # 학생 얼굴 데이터 수집
├── 02_train_model.py           # PyTorch 모델 학습
├── 03_run_attendance.py        # 출석 체크 실행
├── face_model.py               # PyTorch 모델 정의
├── student_manager.py          # 학생 정보 관리
├── haarcascade_frontalface_default.xml  # 얼굴 검출용
├── requirements.txt            # 필요한 패키지 목록
├── dataset/                    # 학생 얼굴 이미지 저장 폴더
├── trainer/                    # 학습된 모델 저장 폴더
│   └── model.pt               # 최종 학습된 모델
├── output/                     # 출석 기록 CSV 파일 저장 폴더
└── students.json               # 학생 정보 (학번: 이름)
```

## 주의사항

1. **학생 등록 순서**: 반드시 `01_collect_data.py` → `02_train_model.py` → `03_run_attendance.py` 순서로 실행하세요.

2. **웹캠 권한**: macOS에서는 웹캠 사용 권한을 허용해야 합니다.

3. **학습 데이터**: 각 학생당 최소 20-30장의 얼굴 이미지가 필요합니다.

4. **조명 조건**: 일정한 조명에서 촬영하고 출석 체크하는 것이 정확도가 높습니다.

## 문제 해결

### 모델 파일을 찾을 수 없다는 오류
- `02_train_model.py`를 먼저 실행하여 모델을 학습하세요.

### 웹캠이 작동하지 않을 때
- 다른 프로그램에서 웹캠을 사용 중인지 확인하세요.
- `cv2.VideoCapture(0)`에서 0 대신 다른 번호(1, 2 등)를 시도해보세요.

### 인식 정확도가 낮을 때
- 더 많은 학습 이미지를 수집하세요.
- 다양한 각도와 표정으로 촬영하세요.
- `03_run_attendance.py`의 `CONFIDENCE_THRESHOLD` 값을 조정하세요 (기본: 0.7).

