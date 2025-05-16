import math
# import cvzone
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
from skimage.morphology import skeletonize
from KalmanFilter_3 import KalmanFilter
import os
from HeadPredict_utils import Process_Predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from LambTracker import LambTracker
from extract_lamb_contour import extract_sheep_contour
from datetime import datetime

print(torch.__version__)
cnt = 0


def Predict(skeleton_points):  # 使用得到的脊椎点进行预测
    """
    对骨架点进行卡尔曼滤波预测
    :param skeleton_points: 列表，包含(x,y)坐标点
    :return: predicted_points: 预测的轨迹点
    """
    # 创建卡尔曼滤波器实例
    kf = KalmanFilter(dt=3.0)
    # 用于存储预测的点
    predicted_points = []

    for i, point in enumerate(skeleton_points):
        current_x, current_y = point

        if i < len(skeleton_points) - 1:
            # 对于除最后一个点外的所有点：更新并预测
            # 1. 首先用当前观测更新滤波器状态
            kf.update(current_x, current_y)

            # 2. 预测下一个位置
            next_x, next_y = kf.predict()
            predicted_points.append((int(next_x), int(next_y)))
        else:
            # 对于最后一个点：更新后再预测下一个可能的位置
            kf.update(current_x, current_y)
            next_x, next_y = kf.predict()
            predicted_points.append((int(next_x), int(next_y)))

    return predicted_points


# def get_skeleton(binary_image):
#     # 使用skimage的骨架化
#     skeleton = skeletonize(binary_image > 0)
#     return skeleton.astype(np.uint8) * 255


# 使用contours中的点寻找小羊的脊椎
def FindSkeleton(contour, image, label, line_color=(0, 255, 0),  # 绘制小羊的脊椎点
                 point_color=(0, 0, 255), line_thickness=1,
                 point_radius=2, x_interval=1, min_line_length=20, sample_num=10,
                 if_fixed_length=True, sample_interval=19):  ## 通过调整采样的间隔来调整卡尔曼滤波算法的速度
    skeleton_points = []
    # 转换为彩色图像
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # 整理点
    points = contour.reshape(-1, 2)
    groups = {}

    # 根据方向选择分组方式
    if label in [0, 2]:  # 'east', 'west'
        # 按 x 坐标分组
        for point in points:
            x = int(point[0] // x_interval) * x_interval  # 离散化 x 坐标
            if x not in groups:
                groups[x] = []
            groups[x].append(point[1])
    elif label in [1, 3]:  # 'north', 'south'
        # 按 y 坐标分组
        for point in points:
            y = int(point[1] // x_interval) * x_interval  # 离散化 y 坐标
            if y not in groups:
                groups[y] = []
            groups[y].append(point[0])

    # 对分组结果进行排序
    if label in [2, 3]:  # 'west', 'south'
        # 从大到小排序
        groups_sorted = dict(sorted(groups.items(), reverse=True))
    else:
        # 从小到大排序
        groups_sorted = dict(sorted(groups.items()))

    # 存储中间过程用于撰写论文
    # back_up = image.copy()
    # cv2.imshow("back", back_up)
    # for key, coords in groups_sorted.items():
    #     if len(coords) >= 2:
    #         coord_min = min(coords)
    #         coord_max = max(coords)
    #         mid_coord = (coord_min + coord_max) // 2
    #
    #         line_length = coord_max - coord_min
    #
    #         start_point = (int(key), int(coord_min))
    #         end_point = (int(key), int(coord_max))
    #         mid_point = (int(key), int(mid_coord))
    #
    #         cv2.line(back_up, start_point, end_point, line_color, line_thickness)
    #         cv2.circle(back_up, mid_point, point_radius, point_color, -1)
    #
    #         cv2.imwrite(f"./test/{key}.jpg", back_up)

    if if_fixed_length:
        # 法一：按照一定的物理距离采样，点的距离保持不变，但数量不固定
        last_selected_key = float('-inf')  # 记录上一个选中的分组键值（x 或 y）
        for key, coords in groups_sorted.items():
            if len(coords) >= 2:
                coord_min = min(coords)
                coord_max = max(coords)
                line_length = coord_max - coord_min  # 计算线段长度

                # 添加间隔判断条件
                if line_length >= min_line_length and abs(key - last_selected_key) >= sample_interval:
                    mid_coord = (coord_min + coord_max) // 2
                    if label in [0, 2]:
                        skeleton_points.append((int(key), int(mid_coord)))
                        start_point = (int(key), int(coord_min))
                        end_point = (int(key), int(coord_max))
                        mid_point = (int(key), int(mid_coord))
                    else:
                        skeleton_points.append((int(mid_coord), int(key)))
                        start_point = (int(coord_min), int(key))
                        end_point = (int(coord_max), int(key))
                        mid_point = (int(mid_coord), int(key))

                    last_selected_key = key  # 更新上一个选中的键值

                    cv2.line(image_color, start_point, end_point, line_color, line_thickness)
                    cv2.circle(image_color, mid_point, point_radius, point_color, -1)
    else:
        # 法二：按照比例采样，采样数量不变
        valid_points = []
        for key, coords in groups_sorted.items():
            if len(coords) >= 2:
                coord_min = min(coords)
                coord_max = max(coords)
                line_length = coord_max - coord_min

                if line_length >= min_line_length:
                    mid_coord = (coord_min + coord_max) // 2
                    if label in [0, 2]:
                        valid_points.append((key, coord_min, coord_max, mid_coord))
                    else:
                        valid_points.append((mid_coord, coord_min, coord_max, key))

        sample_interval = max(1, len(valid_points) // sample_num)  # 取 sample_num 个点
        for i in range(0, len(valid_points), sample_interval):
            if label in [0, 2]:
                key, coord_min, coord_max, mid_coord = valid_points[i]
                skeleton_points.append((int(key), int(mid_coord)))
                start_point = (int(key), int(coord_min))
                end_point = (int(key), int(coord_max))
                mid_point = (int(key), int(mid_coord))
            else:
                mid_coord, coord_min, coord_max, key = valid_points[i]
                skeleton_points.append((int(mid_coord), int(key)))
                start_point = (int(coord_min), int(key))
                end_point = (int(coord_max), int(key))
                mid_point = (int(mid_coord), int(key))

            cv2.line(image_color, start_point, end_point, line_color, line_thickness)
            cv2.circle(image_color, mid_point, point_radius, point_color, -1)

    return skeleton_points, image_color  # 返回脊椎点，以及处理好的图片


def Tailoring(img, y1, y2, x1, x2):
    return img[y1:y2, x1:x2]


def ThresholdImg(img, threshold_value):
    # 将提取图像转化为灰度图
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行阈值化处理
    ret, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_image


def draw_detection_results(img, box, r, contour, x1, y1):
    """
    绘制检测结果，包括边界框、标签和轮廓

    参数:
    img: 要绘制的图像
    box: 检测框对象
    r: 检测结果对象
    contour: 轮廓点集
    x1, y1: 边界框左上角坐标

    返回:
    img_draw: 绘制了检测结果的图像
    """
    img_draw = img.copy()

    # 获取边界框坐标
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 绘制bbox
    conf = float(box.conf)
    cls = int(box.cls)
    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 添加类别标签和置信度
    label = f'{r.names[cls]} {conf:.2f}'
    t_size = cv2.getTextSize(label, 0, fontScale=0.7, thickness=1)[0]
    cv2.rectangle(img_draw, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), (0, 255, 0), -1)
    cv2.putText(img_draw, label, (x1, y1 - 2), 0, 0.7, (0, 0, 0), thickness=1)

    # 将轮廓坐标调整到原图坐标系并绘制
    contour_shifted = contour + [x1, y1]
    cv2.drawContours(img_draw, [contour_shifted], -1, (255, 0, 255), 5)

    return img_draw


def SaveImg(img, file_name, save_dir):  # 保存单个图片
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取文件路径
    file_path = os.path.join(save_dir, file_name)
    base_name, ext = os.path.splitext(file_name)  # 分离文件名和扩展名

    # 检查是否存在重复文件名
    counter = 1
    while os.path.exists(file_path):
        # 如果存在重复文件名，在文件名后添加计数器
        file_name = f"{base_name}_{counter}{ext}"
        file_path = os.path.join(save_dir, file_name)
        counter += 1

    # 保存图片
    if hasattr(img, "save"):  # 如果是 PIL.Image 对象
        img.save(file_path)
    else:  # 如果是 OpenCV 的 numpy 数组
        import cv2
        cv2.imwrite(file_path, img)

    print(f"图片已保存到: {file_path}")


def SaveMidProcessImg(img_list, save_dir="../data/output_images/"):  # 保存所有中间过程的图片
    global cnt
    # 循环保存图片
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(img_list):
        # 定义图片的保存路径和文件名（有序命名）
        file_name = f"frame{cnt}_image_{i + 1}.jpg"  # 使用数字有序命名
        SaveImg(img, file_name, save_dir)

    cnt += 1
    print(f"所有图片已保存到 {save_dir} 文件夹中！")


def fit_and_draw_parabola(skeleton_points, binary_image_color, label, id, mappedHeadP: dict, top_left, frame, degree=1):
    """
    拟合抛物线并将其绘制到图像帧上。
    根据direction决定使用哪个坐标轴方向进行拟合。
    """

    if len(skeleton_points) < 1:
        return
    x, y = zip(*skeleton_points)
    x = np.array(x)
    y = np.array(y)

    if int(label) in [0, 2]:
        # 使用原来的方式，x作为自变量
        X = x.reshape(-1, 1)
        Y = y

        # 创建多项式回归模型
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, Y)

        # 生成拟合曲线的点
        x_fit = np.linspace(min(x) - 20, max(x) + 18, 100)
        y_fit = model.predict(x_fit.reshape(-1, 1))

        # 构建拟合点对
        fit_points = list(zip(x_fit, y_fit))

    else:  # direction in ['north', 'south']
        # 交换xy轴，使用y作为自变量
        X = y.reshape(-1, 1)
        Y = x

        # 创建多项式回归模型
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, Y)

        # 生成拟合曲线的点
        y_fit = np.linspace(min(y), max(y), 100)
        x_fit = model.predict(y_fit.reshape(-1, 1))

        # 构建拟合点对
        fit_points = list(zip(x_fit, y_fit))

    # # 将拟合曲线绘制到binary_image_color上
    # for i in range(1, len(fit_points)):
    #     pt1 = (int(fit_points[i - 1][0]), int(fit_points[i - 1][1]))
    #     pt2 = (int(fit_points[i][0]), int(fit_points[i][1]))
    #     cv2.line(binary_image_color, pt1, pt2, (0, 255, 0), 2)
    #
    # # 绘制原始数据点到binary_image_color
    # for point in skeleton_points:
    #     cv2.circle(binary_image_color, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # 将拟合曲线绘制到frame上
    for i in range(1, len(fit_points)):
        pt1 = (int(fit_points[i - 1][0] + top_left[0]), int(fit_points[i - 1][1] + top_left[1]))
        pt2 = (int(fit_points[i][0] + top_left[0]), int(fit_points[i][1] + top_left[1]))
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    if int(label) == 0:  # east
        end_point = (int(fit_points[-1][0] + top_left[0]), int(fit_points[-1][1] + top_left[1]))
        cv2.circle(frame, end_point, 8, (255, 0, 255), -1)
        mappedHeadP[id] = [int(fit_points[-1][0] + top_left[0]), int(fit_points[-1][1] + top_left[1])]
    elif int(label) == 2:  # east
        end_point = (int(fit_points[0][0] + top_left[0]), int(fit_points[0][1] + top_left[1]))
        cv2.circle(frame, end_point, 8, (255, 0, 255), -1)
        mappedHeadP[id] = [int(fit_points[0][0] + top_left[0]), int(fit_points[0][1] + top_left[1])]
    elif int(label) == 1:  # north
        end_point = (int(fit_points[-1][0] + top_left[0]), int(fit_points[-1][1] + top_left[1]))
        cv2.circle(frame, end_point, 8, (255, 0, 255), -1)
        mappedHeadP[id] = [int(fit_points[-1][0] + top_left[0]), int(fit_points[-1][1] + top_left[1])]
    else:  # south
        end_point = (int(fit_points[0][0] + top_left[0]), int(fit_points[0][1] + top_left[1]))
        cv2.circle(frame, end_point, 8, (255, 0, 255), -1)
        mappedHeadP[id] = [int(fit_points[0][0] + top_left[0]), int(fit_points[0][1] + top_left[1])]

    # # 绘制原始数据点到frame
    # for point in skeleton_points:
    #     cv2.circle(frame, (int(point[0] + top_left[0]), int(point[1] + top_left[1])), 5, (0, 0, 255), -1)


# 版本一，使用检测到的bbox进行处理
def ProcessV1(img, results, direction_model, device):
    global cnt
    mappedHeadP = []

    # 复制原图，避免修改原始图像
    img_draw = img.copy()
    img_list = []
    # 遍历每个检测结果
    boxes = results[0].boxes
    i = 0
    for box in boxes:
        x, y, w, h = box.xywh[0]
        if w * h < 6400:
            continue
        # 获取bbox坐标
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 提取当前bbox区域的图像
        roi = Tailoring(img, y1, y2, x1, x2)
        img_list.append(roi)
        # cv2.imshow("roi", img)
        # 保存裁剪好的图片，用于制作训练集
        # name = f"{cnt}-{i}.jpg"  # 帧编号+编号
        # i += 1
        # SaveImg(roi, name, save_dir="../data/TailoredImg")
        direction_model.eval()
        direction = Process_Predict(img=roi, model=direction_model, device=device)  # 使用模型预测方向

        label = torch.argmax(direction)

        binary_image = ThresholdImg(roi, 120)
        img_list.append(binary_image)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找出最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 转换图像为彩色
        binary_image_color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        # # 绘制点
        # test_image = cv2.circle(binary_image_color, tuple(largest_contour[0][0]), 10, (255, 255, 0), 10)
        # img_list.append(test_image)

        skeleton_points, result_image = FindSkeleton(
            largest_contour,
            binary_image,
            label=label
        )
        img_list.append(result_image)

        predicted_points = Predict(skeleton_points)

        # for p in predicted_points:
        #     mappedHeadP.append((int(p[0] + x1), int(p[1] + y1)))
        p = predicted_points[-1]
        mappedHeadP.append((int(p[0] + x1), int(p[1] + y1)))

        result_image_with_predictions = result_image.copy()

        # for point in predicted_points:
        #     cv2.circle(result_image_with_predictions, point, 4, (255, 0, 0), -1)
        # img_list.append(result_image_with_predictions)
        cv2.imshow("result_image_with_predictions", result_image_with_predictions)
        # SaveImg(result_image_with_predictions, "HeadPrediction.jpg", "../data/HeadPrediciotnImg") #保存使用卡尔曼滤波预测后的图片

        # 使用新的绘制函数
        img_draw = draw_detection_results(img_draw, box, results[0], largest_contour, x1, y1)
        img_list.append(img_draw)

    return img_list, img_draw, result_image_with_predictions, mappedHeadP


# 版本二，使用buffer中的bbox进行处理
def ProcessV2(ori_img, img, results, direction_model, lamb_tracker: LambTracker, device):
    global cnt
    # 复制原图，避免修改原始图像
    img_list = []
    buffer = lamb_tracker.buffer
    back_up = img.copy()
    # 遍历每个检测结果
    # i = 0
    cnt2 = 0
    mappedHeadP = {}  # 存储映射好的头部位置
    for key, box in buffer.items():
        # print(key, box)
        x, y, w, h = box[0], box[1], box[2], box[3]
        if w * h < 6400:
            continue
        # 计算左上角坐标
        x1 = x - w / 2
        y1 = y - h / 2

        # 计算右下角坐标
        x2 = x + w / 2
        y2 = y + h / 2

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # x10, y10, x20, y20 = results[0].boxes[0].xyxy[0]
        # 提取当前bbox区域的图像
        roi = Tailoring(ori_img, y1, y2, x1, x2)
        img_list.append(roi)

        # 保存裁剪好的图片，用于制作训练集
        # name = f"{cnt}-{i}.jpg"  # 帧编号+编号
        # i += 1
        # SaveImg(roi, name, save_dir="../data/TailoredImg")

        direction = Process_Predict(img=roi, model=direction_model, device=device)  # 使用模型预测方向
        direction = torch.softmax(direction, dim=1)  # 对输出进行softmax操作
        label = torch.argmax(direction)

        binary_image = ThresholdImg(roi, 105)

        img_list.append(binary_image)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找出最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 转换图像为彩色
        binary_image_color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        skeleton_points, result_image = FindSkeleton(
            largest_contour,
            binary_image,
            label=label
        )

        top_left = (x1, y1)
        # print(key, direction) #输出预测logits
        # print(key, label)
        fit_and_draw_parabola(skeleton_points, binary_image_color, label, key, mappedHeadP, top_left, img)

        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # cv2.imshow("result_image", binary_image_color)

        # cv2.imwrite(f"./test2/{cnt2}-{timestamp}.jpg", binary_image_color)
        # cnt2 += 1
        img_list.append(result_image)

        # SaveImg(result_image_with_predictions, "HeadPrediction.jpg", "../data/HeadPrediciotnImg") #保存使用卡尔曼滤波预测后的图片

    return img_list, mappedHeadP


if __name__ == "__main__":  # 测试使用
    image_path_1 = "../data/lamb/lamb_keypoints.v2i.coco/train/img_338_jpg.rf.7ab2d2a2436e7250cb1087e0d29e9f6b.jpg"
    image_path_2 = "../data/lamb/lamb_keypoints.v2i.coco/train/img_255_jpg.rf.2fe6767e84bb95e026bf6b396c1871ac.jpg"
    image_path_3 = "../data/lamb/lamb_keypoints.v2i.coco/train/img_347_jpg.rf.afdf3a3b11a95b2d883f5241217cc661.jpg"
    image_path_4 = "../data/lamb/lamb_keypoints.v2i.coco/train/img_271_jpg.rf.800141350ed08240c039cb977c000021.jpg"
    image_path_5 = "../data/north.jpg"

    IMAGE_PATH = image_path_2
    WEIGHTS_PATH = "../weights/lambBody-version2.pt"

    model = YOLO(WEIGHTS_PATH)  # 加载模型
    results = model(IMAGE_PATH)  # 使用模型进行目标检测
    img = cv2.imread(IMAGE_PATH)  # 读取图片

    mid_process_imgs, output_img, img_with_prediciotns = Process(img, results)  # 将预测的信息绘制到图片上
    # SaveMidProcessImg(mid_process_imgs, save_dir="../data/midProcess/") #保存中间过程
    output_img = cv2.resize(output_img, (1080, 640))
    cv2.imshow('Detection with Contours', output_img)
    for idx, mid_process_img in enumerate(mid_process_imgs):
        cv2.imshow("mid process" + str(idx), mid_process_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
