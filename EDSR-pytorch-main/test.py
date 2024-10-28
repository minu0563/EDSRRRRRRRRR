import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from edsr import EDSR  # EDSR 모델이 정의된 파일에서 가져옵니다.

def load_model(model_path, scale_factor):
    model = EDSR(scale_factor)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 모델을 평가 모드로 설정
    return model

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

def process_images(model, lr_image_dir, save_dir):
    # 디렉토리 내 모든 저화질 이미지에 대해 처리
    for img_name in os.listdir(lr_image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일만 처리
            lr_img_path = os.path.join(lr_image_dir, img_name)

            # 저화질 이미지를 업스케일링
            upscaled_image = upscale_image(model, lr_img_path)

            # 업스케일된 이미지 저장 경로 설정
            upscaled_save_path = os.path.join(save_dir, f"upscaled_{img_name}")

            # 업스케일된 이미지 저장
            save_image(upscaled_image, upscaled_save_path)

def main():
    scale_factor = 4  # 스케일 팩터 설정
    model_path = r"C:\Users\User\Desktop\pyth\pytorch-edsr-master\weights\(1)edsr_epoch_10.pth"  # 가중치 경로
    lr_image_dir = r"C:\Users\User\Desktop\pyth\OGQ_LR" # 저화질 이미지 디렉토리
    save_dir = r"C:\Users\User\Desktop\pyth\OGQ_SR"  # 업스케일된 이미지 저장 디렉토리

    # 모델 로드
    model = load_model(model_path, scale_factor)

    # 이미지 처리
    process_images(model, lr_image_dir, save_dir)

if __name__ == '__main__':
    main()
