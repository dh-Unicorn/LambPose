import cv2
import numpy as np


def morphological_operations(binary):
    # 闭运算填充小孔
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 开运算去除噪点
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    return opening


def get_largest_contour(binary):
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 选择最大的轮廓
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None


def extract_sheep_contour(img):
    # 1. 读取和预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 自适应阈值分割
    ret, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. 形态学操作
    morphology = morphological_operations(thresh)

    # 4. 提取最大轮廓
    contour = get_largest_contour(morphology)

    # 5. 绘制结果
    result = img.copy()
    if contour is not None:
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
    cv2.imshow("result", result)
    return contour
