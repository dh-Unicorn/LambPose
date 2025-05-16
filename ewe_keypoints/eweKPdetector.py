import cv2
from ultralytics import YOLO


class eweKPdetector():
    def __init__(self):
        self.model = YOLO("../../weights/ewe_keypoints.pt12/weights/best.pt").to("cuda")

    def detect(self, frame):
        results = self.model(frame)
        return results
