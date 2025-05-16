import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 定义方向对应的类别
DIRECTION_LABELS = {
    "east": [1, 0, 0, 0],
    # "northeast": [0, 1, 0, 0, 0, 0, 0, 0],
    "north": [0, 1, 0, 0],
    # "northwest": [0, 0, 0, 1, 0, 0, 0, 0],
    "west": [0, 0, 1, 0],
    # "southwest": [0, 0, 0, 0, 0, 1, 0, 0],
    "south": [0, 0, 0, 1],
    # "southeast": [0, 0, 0, 0, 0, 0, 0, 1],
}


# 创建文件夹脚本，比手动快多了.命令行才是计算机的正确用法 :）
# for key, value in DIRECTION_LABELS.items():
#     save_path = "../data/HeadPredictionData/val/" + key
#     os.makedirs(save_path, exist_ok=True)


class SheepDirectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集的根目录，例如 './dataset/train'
            transform (callable, optional): 图像预处理变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # 遍历每个方向的文件夹
        for direction, label in DIRECTION_LABELS.items():
            direction_dir = os.path.join(root_dir, direction)
            if not os.path.exists(direction_dir):
                continue
            for file_name in os.listdir(direction_dir):
                file_path = os.path.join(direction_dir, file_name)
                if file_name.endswith(('.jpg', '.png', '.jpeg')):  # 只加载图片文件
                    self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        image = Image.open(file_path).convert("RGB")  # 打开图片并转为 RGB 格式
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
