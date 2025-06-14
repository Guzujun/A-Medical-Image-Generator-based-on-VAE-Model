import os
import torch
import argparse
from pathlib import Path
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random

def process_image(img_path, transform):
    """处理单张图像"""
    img = Image.open(img_path).convert('L')  # 转换为灰度图
    img = transform(img)
    return img

def main(args):
    # 创建保存目录
    save_dir = Path("real")
    save_dir.mkdir(exist_ok=True)
    
    # 设置图像转换
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整为与生成图像相同的尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
    ])
    
    # 收集所有图像路径
    data_dir = Path(args.data_path)
    image_paths = []
    for category in ['NORMAL', 'PNEUMONIA']:
        category_path = data_dir / category
        if category_path.exists():
            image_paths.extend([f for f in category_path.iterdir() 
                              if f.suffix.lower() in ['.jpeg', '.jpg', '.png']])
    
    # 如果图像数量超过需要的数量，随机采样
    if len(image_paths) > args.num_samples:
        image_paths = random.sample(image_paths, args.num_samples)
    else:
        print(f"Warning: Only found {len(image_paths)} images, less than requested {args.num_samples}")
    
    # 处理并保存图像
    print(f"Processing {len(image_paths)} images...")
    for i, img_path in enumerate(tqdm(image_paths)):
        try:
            img = process_image(img_path, transform)
            vutils.save_image(img, save_dir / f"img_{i:05d}.png",
                            normalize=True, value_range=(-1, 1))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Data/chest_xray/train',
                        help='Path to real image directory')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples to process')
    args = parser.parse_args()
    main(args) 