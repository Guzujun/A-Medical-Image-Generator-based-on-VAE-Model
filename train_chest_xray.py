import os
import yaml
import argparse
import numpy as np
from pathlib import Path

import torch
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from experiment import VAEXperiment
from dataset import VAEDataset
from models import *

def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 设置随机种子
    pl.seed_everything(config['exp_params']['manual_seed'])
    
    # 数据集
    data = VAEDataset(**config["data_params"])
    data.setup()
    
    # 创建VAE模型实例
    vae_model = vae_models[config['model_params']['name']](**config['model_params'])
    
    # 创建实验实例
    model = VAEXperiment(vae_model, config['exp_params'])
    
    # 创建日志记录器
    tb_logger = TensorBoardLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
    )
    
    # 创建模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['logging_params']['save_dir'], 
                            config['logging_params']['name'], 
                            'checkpoints'),
        monitor='val_loss',
        mode='min',
        save_top_k=3,
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        **config['trainer_params']
    )
    
    # 开始训练
    trainer.fit(model, data)
    
    # 保存最终模型
    torch.save(model.state_dict(), 
              os.path.join(config['logging_params']['save_dir'],
                          config['logging_params']['name'],
                          'final_model.ckpt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/chest_xray_vae.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    main(args) 