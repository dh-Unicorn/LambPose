from ultralytics import YOLO
from torchvision import transforms
import cv2


def draw_eweKP(frame, results, color=(0, 255, 0), radius=5):
    """
    在图像上绘制 YOLO 检测到的关键点
    :param frame: 输入图像
    :param results: YOLO 检测结果
    :param color: 绘制关键点的颜色
    :param radius: 关键点的半径大小
    """
    for result in results:  # 遍历每个检测结果

        keypoints = result.keypoints.xy.cpu().tolist()
        keypoints = keypoints[0]

        if keypoints is not None:
            for kp in keypoints:  # 遍历每个关键点
                x, y = int(kp[0]), int(kp[1])  # 获取关键点坐标
                cv2.circle(frame, (x, y), radius, color, -1)  # 绘制关键点
    # cv2.imshow("eweKP", frame)


if __name__ == '__main__':
    # 加载训练好的模型
    model = YOLO("../../weights/ewe_keypoints.pt11/weights/best.pt").to("cuda")
    source1 = "E:/@code/@GraduationProject/data/ewe_keyppoints_detection.v1i.yolov8/train/images/img_131_jpg.rf.b31f3516d2f4425308f93d04772da0fe.jpg",
    source2 = "E:/desktop/test.png"
    source3 = "E:/desktop/test2.jpg"
    source4 = "E:/desktop/test3.jpg"

    image = cv2.imread(source4)

    # resized_image = cv2.resize(image, (640, 640))

    # 测试单张图片
    results = model(
        source=image,
        # 测试图片路径
        conf=0.5,  # 设置置信度阈值
        # save=True,  # 保存预测结果
        # save_txt=True,  # 保存预测坐标
    )

    draw_eweKP(results=results, frame=image)

    keypoints_in_pixel = results[0].keypoints.xy.cpu().tolist()
    # print(keypoints_in_pixel)
    # x, y = keypoints_in_pixel[0][2]  # 下标2对于的是头部坐标
    # x, y = keypoints_in_pixel[0][4]  # 下标4对应身体中间位置
    x, y = keypoints_in_pixel[0][5]  # 下标5对应身体尾巴位置
    cv2.circle(image, (int(x), int(y)), radius=2, color=(255, 0, 0), thickness=2)  # 绘制关键点
    image = cv2.resize(image, (1080, 640))
    cv2.imshow("test", image)
    cv2.waitKey(0)
