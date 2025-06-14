import os
import torch
import argparse
from pathlib import Path
import torchvision.utils as vutils
from torchvision import transforms
import yaml
from tqdm import tqdm

from models import *
from experiment import VAEXperiment

def main(args):
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 加载模型
    vae_model = vae_models[config['model_params']['name']](**config['model_params'])
    
    # 加载检查点
    state_dict = torch.load(args.checkpoint)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # 清理state_dict中的'model.'前缀
    cleaned_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    # 加载模型权重
    vae_model.load_state_dict(cleaned_state_dict)
    vae_model.cuda()
    vae_model.eval()
    
    # 创建生成图像保存目录
    save_dir = Path("generated")
    save_dir.mkdir(exist_ok=True)
    
    # 生成样本
    num_samples = args.num_samples
    batch_size = 64
    
    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size)):
            current_batch_size = min(batch_size, num_samples - i)
            # 采样潜在向量
            z = torch.randn(current_batch_size, vae_model.latent_dim).cuda()
            # 生成图像
            samples = vae_model.decode(z)
            # 转换到[0,1]范围
            samples = (samples + 1) / 2
            # 保存图像
            for j, sample in enumerate(samples):
                img_idx = i + j
                # 转换为RGB（如果是灰度图）
                if sample.size(0) == 1:
                    sample = sample.repeat(3, 1, 1)
                # 调整大小到299x299
                # sample = transforms.Resize((256, 256), antialias=True)(sample)
                # 保存为PNG（自动转换为uint8）
                vutils.save_image(sample, save_dir / f"img_{img_idx:05d}.png", 
                                normalize=False, value_range=(0, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/chest_xray_vae.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples to generate')
    args = parser.parse_args()
    main(args) 