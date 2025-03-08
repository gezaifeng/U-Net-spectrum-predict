import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import ResCNN
from dataset import SpectralDataset

# ========================= 训练配置 =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据集路径（请根据实际路径调整）
train_dir = "D:/Desktop/model/Unet Spectrum Model/dataset-76/train"
val_dir   = "D:/Desktop/model/Unet Spectrum Model/dataset-76/val"

# 构建数据集和数据加载器
train_dataset = SpectralDataset(train_dir, normalize_spectra=True, augment=True)
val_dataset   = SpectralDataset(val_dir, normalize_spectra=True, augment=False)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = ResCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 训练超参数
num_epochs = 50
best_val_loss = float("inf")
save_path = "D:/Desktop/model/Unet Spectrum Model/best_model.pth"

# ========================= 训练循环 =========================
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, spectra in train_loader:
        images = images.to(device).float()
        spectra = spectra.to(device).float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, spectra)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    # 验证阶段（不计算梯度）
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, spectra in val_loader:
            images = images.to(device).float()
            spectra = spectra.to(device).float()
            outputs = model(images)
            val_loss += criterion(outputs, spectra).item()
    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 保存最优模型权重
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ 最佳模型更新并保存于 Epoch {epoch+1}，验证损失: {val_loss:.4f}")

print("🎯 训练结束！最佳模型已保存到磁盘。")
