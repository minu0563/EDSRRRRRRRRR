import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from edsr import EDSR
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def load_model(model_path, scale_factor):
    model = EDSR(scale_factor)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 모델을 평가 모드로 설정
    return model

def calculate_metrics(original_image, upscaled_image):
    # PSNR 계산
    psnr = cv2.PSNR(original_image, upscaled_image)

    # SSIM 계산 시 예외 처리
    try:
        height, width, _ = original_image.shape
        win_size = min(height, width, 7)  # 이미지 크기에 맞게 win_size 설정
        ssim_value = ssim(original_image, upscaled_image, channel_axis=2, win_size=win_size)
    except ValueError as e:
        print(f"SSIM 계산 중 오류 발생: {e}")
        ssim_value = None  # SSIM 계산이 안 되면 None으로 설정

    return psnr, ssim_value

def upscale_image(model, img_path):
    # 이미지 로드 및 전처리
    img = Image.open(img_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # 배치 차원 추가

    # 모델에 이미지 입력
    with torch.no_grad():
        output_tensor = model(img_tensor)

    # 텐서를 이미지로 변환
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    output_image = (output_image * 255).clip(0, 255).astype('uint8')  # 값 범위 조정
    return output_image  # 업스케일된 이미지 반환

def save_image(upscaled_image, save_path):
    upscaled_image_pil = Image.fromarray(upscaled_image)
    upscaled_image_pil.save(save_path)
    print(f"업스케일된 이미지가 {save_path}에 저장되었습니다.")

def display_images(original_image, upscaled_image):
    # 두 이미지를 나란히 출력
    plt.figure(figsize=(12, 6))

    # 원본 이미지 출력
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 업스케일된 이미지 출력
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
    plt.title('Upscaled Image')
    plt.axis('off')

    plt.show()

def process_dataset(model, lr_image_dir, hr_image_dir, save_dir):
    # 디렉토리 내 모든 저화질 이미지에 대해 처리
    for img_name in os.listdir(lr_image_dir):
        lr_img_path = os.path.join(lr_image_dir, img_name)
        
        # 원본 고화질 이미지 경로
        hr_img_path = os.path.join(hr_image_dir, img_name)

        # 저화질 이미지를 업스케일링
        upscaled_image = upscale_image(model, lr_img_path)

        # 원본 이미지 로드 (OpenCV 사용)
        original_image = cv2.imread(hr_img_path)
        
        # PSNR 및 SSIM 수치 계산
        psnr, ssim_value = calculate_metrics(original_image, upscaled_image)

        # 업스케일된 이미지 저장 경로 설정
        upscaled_save_path = os.path.join(save_dir, f"upscaled_{img_name}")
        
        # 업스케일된 이미지 저장
        save_image(upscaled_image, upscaled_save_path)

        # PSNR 및 SSIM 출력
        print(f"Image: {img_name} - PSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}")

def main():
    scale_factor = 4  # 스케일 팩터 설정
    model_path = r'C:\Users\User\Desktop\pyth\pytorch-edsr-master\weights\(1)edsr_epoch_10.pth'  # 가중치 경로
    lr_image_dir = r"C:\Users\User\Desktop\pyth\datasets\Set5\LRbicx4"  # 저화질 이미지 디렉토리
    hr_image_dir = r"C:\Users\User\Desktop\pyth\datasets\Set5\original"  # 원본 이미지 디렉토리
    save_dir = r'C:\Users\User\Desktop\pyth\pytorch-edsr-master\UPSCALED_IMAGE'  # 업스케일된 이미지 저장 디렉토리

    # 모델 로드
    model = load_model(model_path, scale_factor)

    # 데이터셋 처리
    process_dataset(model, lr_image_dir, hr_image_dir, save_dir)

if __name__ == '__main__':
    main()
