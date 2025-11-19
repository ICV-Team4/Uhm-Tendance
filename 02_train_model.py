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
import random
import copy
from face_model import FaceRecognitionModel
from student_manager import get_all_students

# --- 설정 ---
IMG_SIZE = 100
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0005  
WEIGHT_DECAY = 1e-4
PATIENCE = 15         
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = 'dataset'
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'   [EarlyStopping] Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss

class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(img_path).convert('L')
            img_numpy = np.array(img, 'uint8')
        except Exception:
            return torch.zeros((1, IMG_SIZE, IMG_SIZE)), label
        
        # 학습 단계에서도 얼굴 영역을 확실하게 다시 잡음
        faces = detector.detectMultiScale(img_numpy, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            face_img = img_numpy[y:y+h, x:x+w]
        else:
            face_img = img_numpy # 얼굴 못 찾으면 전체 이미지 사용
        
        try:
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        except Exception:
             return torch.zeros((1, IMG_SIZE, IMG_SIZE)), label

        face_img = Image.fromarray(face_img)
        
        if self.transform:
            face_img = self.transform(face_img)
        
        return face_img, label

def getImagesAndLabels(path):
    imagePaths = []
    labels = []
    
    if not os.path.exists(path):
        print(f"[ERROR] '{path}' 폴더가 없습니다.")
        return [], []

    print(f"[INFO] 데이터셋 분석 중...")
    for student_id_folder in os.listdir(path):
        student_folder_path = os.path.join(path, student_id_folder)
        if os.path.isdir(student_folder_path):
            try:
                current_id = int(student_id_folder)
            except ValueError:
                continue
            
            count = 0
            for image_file in os.listdir(student_folder_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    imagePaths.append(os.path.join(student_folder_path, image_file))
                    labels.append(current_id)
                    count += 1
            print(f"   -> ID {current_id}: {count} images")
    
    return imagePaths, labels

def train_model():
    print(f"\n[INFO] Device: {DEVICE}")
    
    all_image_paths, all_labels = getImagesAndLabels(path)
    
    if len(all_image_paths) == 0:
        print("[ERROR] No images found!")
        return
    
    unique_ids = sorted(list(set(all_labels)))
    num_classes = len(unique_ids)
    
    id_to_idx = {student_id: idx for idx, student_id in enumerate(unique_ids)}
    idx_to_id = {idx: student_id for student_id, idx in id_to_idx.items()}
    all_label_indices = [id_to_idx[label] for label in all_labels]
    
    # ★★★ [핵심] 클래스 가중치 계산 (데이터 불균형 해결) ★★★
    # 데이터가 적은 클래스는 가중치를 높게, 많은 클래스는 낮게 설정
    class_counts = np.bincount(all_label_indices)
    total_samples = len(all_label_indices)
    
    # 가중치 공식: Total / (Num_Classes * Class_Count)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    
    print(f"\n[INFO] Class Weights (불균형 보정):")
    for idx, weight in enumerate(class_weights):
        print(f"   ID {idx_to_id[idx]}: {weight:.4f} (Count: {class_counts[idx]})")

    # Train/Val Split
    combined = list(zip(all_image_paths, all_label_indices))
    random.shuffle(combined)
    all_image_paths[:], all_label_indices[:] = zip(*combined)
    
    split_idx = int(len(all_image_paths) * 0.8)
    train_paths, val_paths = all_image_paths[:split_idx], all_image_paths[split_idx:]
    train_labels, val_labels = all_label_indices[:split_idx], all_label_indices[split_idx:]
    
    # 강력한 데이터 증강
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # 위치 이동 추가
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = FaceDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = FaceDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = FaceRecognitionModel(num_classes).to(DEVICE)
    
    # ★★★ 손실 함수에 가중치 적용 ★★★
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n[INFO] Start Training ({EPOCHS} epochs)...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, target_labels in train_loader:
            images, target_labels = images.to(DEVICE), target_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, target_labels in val_loader:
                images, target_labels = images.to(DEVICE), target_labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, target_labels)
                val_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target_labels.size(0)
                val_correct += (predicted == target_labels).sum().item()
        
        val_loss = val_loss_sum / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss:{train_loss:.4f} Acc:{train_acc:.1f}% | Val Loss:{val_loss:.4f} Acc:{val_acc:.1f}%")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("[INFO] Early stopping!")
            break
    
    # 모델 저장
    model.load_state_dict(early_stopping.best_model_state)
    trainer_folder = 'trainer'
    if not os.path.exists(trainer_folder): os.makedirs(trainer_folder)
    
    model_path = f'{trainer_folder}/model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'idx_to_id': idx_to_id,
        'id_to_idx': id_to_idx,
        'img_size': IMG_SIZE
    }, model_path)
    
    print(f"\n[INFO] Weighted Model saved: {model_path}")
    
    # 결과 시각화 (학습 후 그래프 확인용)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plot_path = 'training_history/result.png'
    plt.savefig(plot_path)
    print(f"[INFO] Result plot saved: {plot_path}")

if __name__ == "__main__":
    train_model()