import torch
import torch.nn as nn
import torch.nn.functional as F


# README！
# 该文件用于绘制模型的结构图，但要求环境的Python版本大于3.10。否则无法正常运行！
# 生成出图像极为模糊，只供参考，建议手绘。

class DirectionClassifier(nn.Module):
    def __init__(self, input_size=64):
        super(DirectionClassifier, self).__init__()
        self.input_size = input_size

        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 第三层卷积
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Dropout 层
        self.dropout = nn.Dropout(0.5)

        # 全连接层
        fc_input_size = (input_size // 8) * (input_size // 8) * 64
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 8)  # 输出 8 个方向的概率值

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # 使用 Sigmoid 激活函数
        return x


import matplotlib.pyplot as plt
import visualtorch
from torch import nn

input_shape = (1, 3, 64, 64)

model = DirectionClassifier()

img = visualtorch.lenet_view(model, input_shape=input_shape)
fig = plt.figure(figsize=(12, 12))
plt.axis("off")
plt.tight_layout()
plt.imshow(img)
plt.savefig('lenet_structure.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.show()
