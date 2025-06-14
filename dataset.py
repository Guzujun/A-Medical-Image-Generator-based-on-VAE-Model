import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import PIL.Image as Image


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class ChestXrayDataset(Dataset):
    """
    Dataset class for Chest X-ray images
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                 **kwargs):
        self.data_dir = Path(data_path) / split
        self.transforms = transform
        
        # 获取所有图像文件
        self.imgs = []
        for category in ['NORMAL', 'PNEUMONIA']:
            category_path = self.data_dir / category
            if category_path.exists():
                self.imgs.extend([f for f in category_path.iterdir() if f.suffix.lower() in ['.jpeg', '.jpg', '.png']])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # 使用PIL加载图像并转换为灰度
        img = Image.open(img_path).convert('L')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0  # 0.0是dummy label，VAE不需要标签

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        # 定义数据转换
        train_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        eval_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # 创建训练集、验证集和测试集
        self.train_dataset = ChestXrayDataset(
            self.data_dir,
            split='train',
            transform=train_transforms,
        )
        
        self.val_dataset = ChestXrayDataset(
            self.data_dir,
            split='val',
            transform=eval_transforms,
        )
        
        self.test_dataset = ChestXrayDataset(
            self.data_dir,
            split='test',
            transform=eval_transforms,
        )
        
        # 打印数据集大小信息
        print(f"Dataset sizes:")
        print(f"Train: {len(self.train_dataset)}")
        print(f"Val: {len(self.val_dataset)}")
        print(f"Test: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,  # 使用测试集而不是验证集
            batch_size=self.val_batch_size,  # 使用相同的batch_size
            num_workers=self.num_workers,
            shuffle=False,  # 评估时不需要shuffle
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
     