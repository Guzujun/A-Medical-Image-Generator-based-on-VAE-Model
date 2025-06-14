import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.cached_test_batch = None  # 添加缓存
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'])

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # 使用缓存的测试数据或首次加载
        if self.cached_test_batch is None:
            test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
            self.cached_test_batch = (test_input, test_label)
        else:
            test_input, test_label = self.cached_test_batch
            
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons = self.model.generate(test_input, labels = test_label)
        
        # 创建保存重构图像的目录
        recons_dir = os.path.join(self.logger.log_dir, "Reconstructions")
        os.makedirs(recons_dir, exist_ok=True)
        
        vutils.save_image(recons.data,
                          os.path.join(recons_dir, 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            
            # 创建保存生成图像的目录
            samples_dir = os.path.join(self.logger.log_dir, "Samples")
            os.makedirs(samples_dir, exist_ok=True)
            
            vutils.save_image(samples.cpu().data,
                              os.path.join(samples_dir,      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        
        scheduler = None
        if self.params.get('scheduler_gamma') is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                         gamma = self.params['scheduler_gamma'])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }
        
        return optimizer
