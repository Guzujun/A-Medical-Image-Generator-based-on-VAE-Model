import os
import torch
import argparse
from pathlib import Path
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torchmetrics.image.fid as fid
import torchmetrics.image.inception as inception_score
from torch.utils.data import Dataset, DataLoader

def to_uint8_tensor(x):
    """将张量转换为uint8类型"""
    return (x * 255).byte()

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        self.image_files = sorted([f for f in self.folder_path.iterdir() 
                                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # 加载为RGB图像
        img = Image.open(img_path).convert('RGB')
        # 调整大小到299x299（Inception V3的输入尺寸）
        img = img.resize((299, 299), Image.Resampling.LANCZOS)
        # 转换为uint8张量
        if self.transform:
            img = self.transform(img)
        return img

def load_images(folder_path, batch_size=32):
    """加载图像文件夹"""
    # 只转换为张量，保持uint8类型
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_uint8_tensor)  # 使用命名函数替代lambda
    ])
    
    dataset = ImageFolderDataset(folder_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4)
    
    images = []
    for batch in tqdm(dataloader, desc=f"Loading {folder_path}"):
        images.append(batch)
    return torch.cat(images, dim=0)

def compute_metrics(real_path, generated_path):
    """计算FID和Inception Score"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载图像
    print("Loading images...")
    real_imgs = load_images(real_path)
    generated_imgs = load_images(generated_path)
    
    print(f"Loaded {len(real_imgs)} real images and {len(generated_imgs)} generated images")
    print(f"Image type: {real_imgs.dtype}, shape: {real_imgs.shape}")
    print(f"Value range: [{real_imgs.min()}, {real_imgs.max()}]")
    
    # 计算FID
    print("Computing FID score...")
    fid_metric = fid.FrechetInceptionDistance(normalize=True)
    fid_metric = fid_metric.to(device)
    
    # 分批处理以节省内存
    batch_size = 32
    for i in range(0, len(real_imgs), batch_size):
        batch = real_imgs[i:i+batch_size].to(device)
        fid_metric.update(batch, real=True)
    
    for i in range(0, len(generated_imgs), batch_size):
        batch = generated_imgs[i:i+batch_size].to(device)
        fid_metric.update(batch, real=False)
    
    fid_score = float(fid_metric.compute())
    print(f"FID Score: {fid_score:.4f}")
    
    # 计算Inception Score
    print("Computing Inception Score...")
    is_metric = inception_score.InceptionScore()
    is_metric = is_metric.to(device)
    
    for i in range(0, len(generated_imgs), batch_size):
        batch = generated_imgs[i:i+batch_size].to(device)
        is_metric.update(batch)
    
    is_mean, is_std = is_metric.compute()
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
    
    # 保存结果
    results_file = "evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write(f"FID Score: {fid_score:.4f}\n")
        f.write(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}\n")
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_path', type=str, default='real',
                        help='Path to real images directory')
    parser.add_argument('--generated_path', type=str, default='generated',
                        help='Path to generated images directory')
    args = parser.parse_args()
    compute_metrics(args.real_path, args.generated_path) 