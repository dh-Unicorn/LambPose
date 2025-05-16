import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt


def get_skeleton(binary_image):
    # 使用skimage的骨架化
    skeleton = skeletonize(binary_image > 0)
    return skeleton.astype(np.uint8) * 255


def get_medial_axis(binary_image):
    # 计算距离变换
    dist_transform = distance_transform_edt(binary_image)

    # 计算局部最大值作为中轴
    local_max = peak_local_max(dist_transform, indices=False, labels=binary_image)
    return local_max.astype(np.uint8) * 255


def process_contour_image(img):
    if img is None:
        # 创建示例图像（羊的简化轮廓）
        img = np.zeros((400, 600), dtype=np.uint8)
        # 画一个简单的羊形状
        pts = np.array([[200, 200], [300, 150], [400, 200], [350, 300], [250, 300]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], 255)

    # 确保图像是二值图像
    _, binary = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)

    # 获取骨架
    skeleton = get_skeleton(binary)

    # 显示结果
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(binary, cmap='gray')
    plt.title('Original Binary Image')

    plt.subplot(132)
    plt.imshow(skeleton, cmap='gray')
    plt.title('Skeleton')

    # # 将骨架叠加到原图上
    # overlay = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    # skeleton_points = np.where(skeleton > 0)
    # overlay[skeleton_points[0], skeleton_points[1]] = [0, 0, 255]  # 红色显示骨架
    #
    # plt.subplot(133)
    # plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    # plt.title('Overlay')
    #
    # plt.tight_layout()
    plt.show()
