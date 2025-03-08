import os
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image

# ========================= 数据集定义 =========================
class SpectralDataset(Dataset):
    def __init__(self, data_dir, normalize_spectra=True, augment=False):
        """
        读取存储在 data_dir 目录下的 RGB 和光谱数据，并进行必要的预处理
        :param data_dir: 数据集文件夹路径
        :param normalize_spectra: 是否对光谱数据进行归一化
        :param augment: 是否对 RGB 数据进行数据增强（仅在训练集上使用）
        """
        self.data_dir = data_dir
        self.rgb_files = sorted([f for f in os.listdir(data_dir)
                                 if f.startswith("rgb") and f.endswith(".npy")])
        self.spectral_files = [f.replace("rgb_", "spectral_")
                                for f in self.rgb_files]

        self.normalize_spectra = normalize_spectra
        self.augment = augment
        self.mean = None
        self.std = None

        # 计算光谱数据的均值和标准差（用于归一化）
        if self.normalize_spectra:
            self.mean, self.std = self.compute_spectra_stats()

    def compute_spectra_stats(self):
        """计算所有光谱数据的均值和标准差"""
        all_spectra = []
        for spectral_file in self.spectral_files:
            spectral_path = os.path.join(self.data_dir, spectral_file)
            spectral_data = np.load(spectral_path).astype(np.float32)
            all_spectra.append(spectral_data)
        all_spectra = np.stack(all_spectra)  # (N, 76) 假设光谱长度为 76
        return np.mean(all_spectra, axis=0), np.std(all_spectra, axis=0)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # 读取 RGB 和光谱数据
        rgb_path = os.path.join(self.data_dir, self.rgb_files[idx])
        spectral_path = os.path.join(self.data_dir, self.spectral_files[idx])
        rgb_data = np.load(rgb_path).astype(np.float32)   # 原始形状 (4, 6, 100, 3)
        spectral_data = np.load(spectral_path).astype(np.float32)  # 光谱形状 (76,)

        # 归一化 RGB 数据到 [0,1]
        rgb_data /= 255.0

        # 对 100 个随机点取均值 (mean pooling)，得到每个空间位置的 RGB 平均值 (4, 6, 3)
        rgb_mean = np.mean(rgb_data, axis=2)  # 形状变为 (4, 6, 3)

        # 归一化光谱数据（基于整个数据集的均值和标准差）
        if self.normalize_spectra:
            spectral_data = (spectral_data - self.mean) / (self.std + 1e-8)  # 加小量避免除零

        # 数据增强（仅在训练集上进行）
        if self.augment:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop((4, 6), scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                # 添加高斯噪声并裁剪到 [0,1]
                transforms.Lambda(lambda img: (img + torch.randn_like(img) * 0.05).clamp(0.0, 1.0))
            ])
            # 注意：transforms.ToPILImage() 需要 uint8 格式，确保转换前乘以 255 后转换为 uint8
            rgb_aug = (rgb_mean * 255.0).astype(np.uint8)  # [0,255] uint8
            rgb_tensor = transform(rgb_aug)  # 形状 (3, 4, 6)，值域 [0,1]
        else:
            # 无数据增强时，直接转为张量并调整维度为 (C, H, W) = (3, 4, 6)
            rgb_tensor = torch.tensor(rgb_mean, dtype=torch.float32).permute(2, 0, 1)  # (3,4,6)

        spectral_tensor = torch.tensor(spectral_data, dtype=torch.float32)  # (76,)
        return rgb_tensor, spectral_tensor
