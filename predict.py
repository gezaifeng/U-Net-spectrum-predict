import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 1️⃣ 确保 Matplotlib 运行在交互模式
matplotlib.use("TkAgg")  # 或 'Qt5Agg'，保证 GUI 支持
plt.ion()  # 进入 Matplotlib 交互模式（Interactive Mode）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体 (SimHei) 显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号无法显示问题
from model import ResCNN
from dataset import SpectralDataset

# ========================= 模型加载 =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "D:/Desktop/model/Unet Spectrum Model/best_model.pth"

# 初始化模型并加载训练好的权重
model = ResCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 加载训练数据集以获取其均值和标准差（用于预测时的反归一化）
train_dir = "D:/Desktop/model/Unet Spectrum Model/dataset-76/train"
train_dataset = SpectralDataset(train_dir, normalize_spectra=True, augment=False)


def visualize_prediction(true_spectrum, predicted_spectrum, wavelengths=None):
    """
    可视化 真实光谱 vs 预测光谱
    """
    if wavelengths is None:
        wavelengths = np.arange(380, 760, 5)

    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, true_spectrum, 'bo-', label='real spectrum')
    plt.plot(wavelengths, predicted_spectrum, 'ro-', label='predict spectrum')

    plt.xlabel("wavelength")
    plt.ylabel("absorbance")
    plt.title("Spectral prediction results")
    plt.legend()
    plt.grid()
    plt.show(block=True)  # 强制 Matplotlib 阻塞运行，等待窗口关闭
def predict(rgb_npy_path, dataset):
    """
    预测单个 RGB 数据对应的光谱数据
    :param rgb_npy_path: 待预测的 RGB 数据文件路径 (形状 4*6*100*3)
    :param dataset: 训练使用的 SpectralDataset，用于归一化参数
    :return: 预测的光谱值（已反归一化到原始尺度）
    """
    # 加载 RGB 数据并预处理
    rgb_data = np.load(rgb_npy_path).astype(np.float32)   # (4, 6, 100, 3)
    rgb_data /= 255.0
    rgb_mean = np.mean(rgb_data, axis=2)                  # (4, 6, 3) 取平均得到RGB统计值
    rgb_tensor = torch.tensor(rgb_mean, dtype=torch.float32).permute(2, 0, 1)  # (3,4,6)
    rgb_tensor = rgb_tensor.unsqueeze(0).to(device)       # 添加 batch 维度 -> (1,3,4,6)

    # 执行预测
    with torch.no_grad():
        pred_spectrum_norm = model(rgb_tensor).cpu().numpy().flatten()  # (76,) 归一化光谱
    # 将光谱从归一化恢复到真实尺度
    if dataset.normalize_spectra:
        pred_spectrum = pred_spectrum_norm * (dataset.std + 1e-8) + dataset.mean
    else:
        pred_spectrum = pred_spectrum_norm
    return pred_spectrum

# ========================= 示例预测 =========================
# 请将 test_rgb_path 替换为实际的待预测 RGB .npy 文件路径
test_rgb_path = "D:/Desktop/model/Unet Spectrum Model/dataset-76/val/rgb_0292.npy"
true_spectrum_path = "D:/Desktop/model/Unet Spectrum Model/dataset-76/val/spectral_0292.npy"

predicted_spectrum = predict(test_rgb_path, train_dataset)
true_spectrum = np.load(true_spectrum_path).astype(np.float32)

print("预测的光谱数据:", predicted_spectrum)
# 可视化真实 vs 预测光谱
visualize_prediction(true_spectrum, predicted_spectrum)
