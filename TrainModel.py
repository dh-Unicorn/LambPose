import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dataloader import SheepDirectionDataset
from PredictBodyDirection import DirectionClassifier
from torch import nn


# 训练函数
def train_model(model, train_loader, val_loader, lossFunction, optimizer, num_epochs=20, device='cpu'):
    model.to(device)
    best_model_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = lossFunction(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

        # 验证模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs, dim=1)  # 获取概率最大的方向
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")
        if correct / total > best_model_acc:
            best_model_acc = correct / total
            save_path = "../weights/best_HeadPrediction.pt"  # 保存的文件路径
            torch.save(model.state_dict(), save_path)
    print("Training complete!")


# 主程序
def main():
    train_dir = '../data/BodyPredictionData/train'
    val_dir = '../data/BodyPredictionData/val'

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 创建数据集和数据加载器
    train_dataset = SheepDirectionDataset(train_dir, transform=transform)
    val_dataset = SheepDirectionDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 检查数据加载是否正常
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break

    # 初始化模型、损失函数和优化器
    model = DirectionClassifier(input_size=128)
    lossFunction = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练模型
    train_model(model, train_loader, val_loader, lossFunction, optimizer, num_epochs=200, device=device)

    save_path = "../weights/HeadPrediction.pt"  # 保存的文件路径
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")


if __name__ == "__main__":
    main()
