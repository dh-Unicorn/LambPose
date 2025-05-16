from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8m-pose.pt").to("cuda")

    # Train the model
    results = model.train(
        data="E:\desktop\ewe_keyppoints_detection.v2i.yolov8-4.10\data.yaml",
        epochs=100,
        project="../../weights",
        name="ewe_keypoints.pt"
    )
