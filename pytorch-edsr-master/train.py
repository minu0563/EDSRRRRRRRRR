import os
import time  # 시간을 측정하기 위한 모듈
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from edsr import EDSR  # EDSR 모델 파일에서 Net 클래스를 가져옵니다.
from tqdm import tqdm  # tqdm 라이브러리로 프로그레스 바를 생성

# 데이터셋 클래스 정의
class DIV2KDataset(Dataset):
    def __init__(self, images_dir, scale):
        self.images_dir = images_dir
        self.scale = scale
        self.image_names = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        self.fixed_size = (512, 512)  # 메모리 절약을 위해 크기를 줄임

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        img_path = os.path.join(self.images_dir, image_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.fixed_size, Image.BICUBIC)
        lr_img = img.resize((img.width // self.scale, img.height // self.scale), Image.BICUBIC)

        hr_tensor = self.transform(img)
        lr_tensor = self.transform(lr_img)

        return lr_tensor, hr_tensor

# 모델 학습 함수 정의
def train_model(data_loader, num_epochs=10, learning_rate=1e-4, save_dir=r"C:\Users\User\Desktop\pyth\pytorch-edsr-master\weights"):
    scale_factor = 2  # 스케일 팩터를 정의합니다.
    model = EDSR(scale_factor)  # EDSR 모델 초기화
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # CUDA가 사용 가능한지 확인하고 장치를 설정합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 모델을 선택된 장치로 이동
    criterion = criterion.to(device)  # 손실 함수를 선택된 장치로 이동

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 가중치를 저장할 경로가 없다면 생성

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()  # 에포크 시작 시간 기록
        epoch_loss = 0  # 에포크의 전체 손실을 저장할 변수

        # tqdm을 사용한 프로그레스 바 설정
        with tqdm(total=len(data_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]") as pbar:
            for iteration, (lr_imgs, hr_imgs) in enumerate(data_loader):
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)  # 데이터를 선택된 장치로 이동

                optimizer.zero_grad()
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 프로그레스 바 업데이트
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        # 에포크가 끝날 때의 시간을 기록하고 전체 걸린 시간 출력
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - start_time

        # 각 에포크의 평균 손실과 걸린 시간 출력
        print(f"Epoch [{epoch + 1}/{num_epochs}] 완료! 평균 Loss: {epoch_loss / len(data_loader):.4f}, Epoch Time: {epoch_duration:.2f} seconds")

        # 매 에포크마다 학습된 가중치를 저장
        torch.save(model.state_dict(), os.path.join(save_dir, f"edsr_epoch_{epoch + 1}.pth"))
        print(f"Epoch [{epoch + 1}/{num_epochs}]의 가중치 저장 완료!")

# 경로 설정
images_dir = r"C:\Users\User\Desktop\pyth\DIV2K\DIV2K_train_HR"  # 경로를 적절하게 변경하세요.
scale = 2
dataset = DIV2KDataset(images_dir=images_dir, scale=scale)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 모델 학습
train_model(data_loader, num_epochs=10, save_dir=r"C:\Users\User\Desktop\pyth\pytorch-edsr-master\weights")