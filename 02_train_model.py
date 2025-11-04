import cv2
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import datetime  
from face_model import FaceRecognitionModel
from student_manager import get_all_students

# 설정
IMG_SIZE = 100  # 얼굴 이미지 리사이즈 크기
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = 'dataset'
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class FaceDataset(Dataset):
    """얼굴 이미지 데이터셋"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        img = Image.open(img_path).convert('L')
        img_numpy = np.array(img, 'uint8')
        
        # 얼굴 검출
        faces = detector.detectMultiScale(img_numpy)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = img_numpy[y:y+h, x:x+w]
        else:
            face_img = img_numpy
        
        # 리사이즈
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        
        # PIL Image로 변환
        face_img = Image.fromarray(face_img)
        
        if self.transform:
            face_img = self.transform(face_img)
        else:
            # 기본 변환: Tensor로 변환 및 정규화
            face_img = transforms.ToTensor()(face_img)
        
        return face_img, label

def getImagesAndLabels(path):
    """데이터셋에서 이미지 경로와 레이블 추출 (dataset/{id} 구조 지원)"""
    imagePaths = []
    labels = []
    
    print(f"[INFO] Reading images from base folder: {path}")

    # 'path'(dataset) 폴더 내의 모든 하위 폴더(학생 ID)를 순회합니다.
    for student_id_folder in os.listdir(path):
        student_folder_path = os.path.join(path, student_id_folder)
        
        # 하위 경로가 폴더인지 확인합니다.
        if os.path.isdir(student_folder_path):
            try:
                # 폴더 이름을 정수형 ID로 변환합니다.
                current_id = int(student_id_folder)
            except ValueError:
                # 폴더 이름이 숫자가 아니면 (예: .DS_Store) 건너뜁니다.
                print(f"[WARN] Skipping non-numeric folder: {student_id_folder}")
                continue
                
            # 해당 학생 폴더 내의 모든 .jpg 파일을 순회합니다.
            for image_file in os.listdir(student_folder_path):
                if image_file.endswith('.jpg'):
                    # 전체 이미지 경로를 리스트에 추가합니다.
                    imagePaths.append(os.path.join(student_folder_path, image_file))
                    # 해당 ID(레이블)를 리스트에 추가합니다.
                    labels.append(current_id)
    
    return imagePaths, labels

def train_model():
    """PyTorch 모델 학습"""
    print(f"\n[INFO] Using device: {DEVICE}")
    print("[INFO] Loading dataset...")
    
    image_paths, labels = getImagesAndLabels(path)
    
    if len(image_paths) == 0:
        print("[ERROR] No images found in dataset folder!")
        return
    
    # 고유한 학생 ID 수 확인
    unique_ids = sorted(list(set(labels)))
    num_classes = len(unique_ids)
    
    # ID를 0부터 시작하는 인덱스로 매핑
    id_to_idx = {student_id: idx for idx, student_id in enumerate(unique_ids)}
    idx_to_id = {idx: student_id for student_id, idx in id_to_idx.items()}
    
    # 레이블을 인덱스로 변환
    label_indices = [id_to_idx[label] for label in labels]
    
    print(f"[INFO] Found {len(image_paths)} images from {num_classes} students")
    print(f"[INFO] Student IDs: {unique_ids}")
    
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # -1 ~ 1 범위로 정규화
    ])
    
    # 데이터셋 및 DataLoader 생성
    dataset = FaceDataset(image_paths, label_indices, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 모델 초기화
    model = FaceRecognitionModel(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    # 에폭별 손실과 정확도를 기록할 리스트
    history_loss = []
    history_acc = []
    
    print(f"\n[INFO] Training model for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, target_labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            target_labels = target_labels.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # 통계
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
            
        scheduler.step()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        # 현재 에폭의 결과 기록
        history_loss.append(epoch_loss)
        history_acc.append(epoch_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # 파일명에 사용할 현재 시간 생성
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S") # YYYYMMDD_HHMMSS
    
    # 모델 저장
    trainer_folder = 'trainer'
    if not os.path.exists(trainer_folder):
        os.makedirs(trainer_folder)
    
    model_path = f'{trainer_folder}/model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'idx_to_id': idx_to_id,
        'id_to_idx': id_to_idx,
        'img_size': IMG_SIZE
    }, model_path)
    
    print(f"\n[INFO] Model saved to: {model_path}")

    # 학습 완료 후 그래프 생성 및 저장
    print("\n[INFO] Generating training plot...")
    plt.figure(figsize=(12, 5))
    
    # 에폭 범위 (1부터 시작)
    epochs_range = range(1, EPOCHS + 1)
    
    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_loss, label='Training Loss', marker='o', linestyle='-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_acc, label='Training Accuracy', marker='o', linestyle='-', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 플롯 저장 폴더 생성
    plot_folder = 'training_history'
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    plot_path = f'{plot_folder}/training_plot_{timestamp}.png'
    plt.savefig(plot_path)
    print(f"[INFO] Training plot saved to: {plot_path}")
    
    print(f"[INFO] Trained {num_classes} classes successfully!")

if __name__ == "__main__":
    train_model()