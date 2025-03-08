import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================= U-Net 模型定义（ResCNN命名） =========================
class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()
        # 输入通道为 3（RGB），采用 U-Net 编码器结构
        # 编码器（下采样路径）
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # 输出 (32,4,6)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 输出 (64,4,6)
        self.pool1 = nn.MaxPool2d(2, 2)                                # 下采样至 (64,2,3)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输出 (128,2,3)
        self.enc_conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # 输出 (128,2,3)
        self.pool2 = nn.MaxPool2d(2, 2)                               # 下采样至 (128,1,1)

        # 底部瓶颈层（无需进一步下采样）
        self.bottom_conv = nn.Conv2d(128, 128, kernel_size=1)         # 输出 (128,1,1)

        # 解码器（上采样路径）
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(128+128, 128, kernel_size=3, padding=1)  # Skip 连接拼接 (128↓ + 128↑)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, output_padding=(0,1))
        self.dec_conv3 = nn.Conv2d(64+64, 64, kernel_size=3, padding=1)    # Skip 连接拼接 (64↓ + 64↑)
        self.dec_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # 光谱预测的全连接层
        self.fc1 = nn.Linear(32 * 4 * 6, 256)   # 平坦化大小：32 通道 * 4 * 6 网格
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 76)          # 输出 76 维光谱值

    def forward(self, x):
        # 编码阶段
        e1 = F.relu(self.enc_conv1(x))  # (batch, 32, 4, 6)
        e2 = F.relu(self.enc_conv2(e1))  # (batch, 64, 4, 6)
        x_pool1 = self.pool1(e2)  # (batch, 64, 2, 3)
        e3 = F.relu(self.enc_conv3(x_pool1))  # (batch, 128, 2, 3)
        e4 = F.relu(self.enc_conv4(e3))  # (batch, 128, 2, 3)
        x_pool2 = self.pool2(e4)  # (batch, 128, 1, 1)

        # 底部瓶颈层
        btm = F.relu(self.bottom_conv(x_pool2))  # (batch, 128, 1, 1)

        # 解码阶段 + 跳跃连接
        d1 = self.upconv1(btm)  # 上采样至 (batch, 128, 2, 4)
        d1 = self.crop_and_concat(d1, e4)  # 确保拼接前尺寸匹配 (batch, 256, 2, 3)
        d1 = F.relu(self.dec_conv1(d1))  # (batch, 128, 2, 3)
        d2 = F.relu(self.dec_conv2(d1))  # (batch, 64, 2, 3)

        d3 = self.upconv2(d2)  # 上采样至 (batch, 64, 4, 6)
        d3 = self.crop_and_concat(d3, e2)  # 确保拼接前尺寸匹配 (batch, 128, 4, 6)
        d3 = F.relu(self.dec_conv3(d3))  # (batch, 64, 4, 6)
        d4 = F.relu(self.dec_conv4(d3))  # (batch, 32, 4, 6)

        # 展平 + 全连接层
        out = d4.view(d4.size(0), -1)  # (batch_size, 32*4*6)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)  # (batch_size, 76)
        return out

    def crop_and_concat(self, upsampled, downsampled):
        """
        修正拼接时的维度不匹配问题：
        - 如果 `upsampled` 的 H/W 过大，则进行裁剪
        - 如果 `upsampled` 的 H/W 过小，则使用 `F.interpolate()` 进行插值
        """
        _, _, h1, w1 = upsampled.shape
        _, _, h2, w2 = downsampled.shape

        # 1️⃣  如果 upsampled 尺寸大于 downsampled，则裁剪
        if h1 > h2:
            upsampled = upsampled[:, :, :h2, :]
        if w1 > w2:
            upsampled = upsampled[:, :, :, :w2]

        # 2️⃣  如果 upsampled 尺寸小于 downsampled，则插值到匹配大小
        if h1 < h2 or w1 < w2:
            upsampled = F.interpolate(upsampled, size=(h2, w2), mode="bilinear", align_corners=False)

        return torch.cat([upsampled, downsampled], dim=1)  # 进行拼接


