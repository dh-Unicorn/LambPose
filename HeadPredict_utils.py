import cv2
import torch
from PredictBodyDirection import DirectionClassifier
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

import torch
import torchvision.transforms as transforms


def Process_Predict(model, img, device):
    """
    将 NumPy 图像数组转换为 PyTorch 张量并传递给模型进行预测。

    参数:
    - model: PyTorch 模型
    - img: NumPy 图像数组，通常形状为 (H, W, C)
    - device: 设备（'cpu' 或 'cuda'）

    返回:
    - direction: 模型预测结果
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 将 NumPy 数组或 PIL 图像转换为 PyTorch 张量，并将像素值归一化到 [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 标准化到 [-1, 1]
        transforms.Resize((128, 128))
    ])

    # 如果输入图像是 NumPy 数组，确保它是 (H, W, C) 格式
    if len(img.shape) == 2:  # 如果是灰度图，扩展为 (H, W, 1)
        img = img[:, :, None]
    elif img.shape[0] == 3:  # 如果是 (C, H, W)，需要转为 (H, W, C)
        img = img.transpose(1, 2, 0)

    img_tensor = preprocess(img)  # 输出形状为 (C, H, W)

    # 添加 batch 维度，形状变为 (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)

    img_tensor = img_tensor.to(device)

    # 设置模型为评估模式
    model.eval()

    # 禁用梯度计算（提高推理效率）
    with torch.no_grad():
        # 将图像输入模型并获取预测结果
        direction = model(img_tensor)

    return direction
