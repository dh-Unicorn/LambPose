import cv2
import torch
from PredictBodyDirection import DirectionClassifier
from torchvision import transforms
from PIL import Image

weightPath = "../weights/HeadPrediction.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
direction_model = DirectionClassifier()
direction_model.load_state_dict(torch.load(weightPath))
direction_model.to(device)

testfile = "../data/BodyPredictionData/val/east/1-1.jpg"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 加载图片
image = Image.open(testfile).convert("RGB")  # 确保图片是 RGB 模式

transformed_image = transform(image).unsqueeze(dim=0).to(device)
print(transformed_image)
pred = direction_model(transformed_image)

result = torch.argmax(pred)
print(result)
