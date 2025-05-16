from collections import defaultdict
from time import sleep
import cvzone
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.cm as cm
from PIL import Image, ImageTk
import os
import time


class LambTracker:
    def __init__(self):
        self.total_num = 4
        self.buffer = {}
        self.buffer_origin = {}
        self.stand = [0] * 10  # 站立状态
        self.lickedTimes = [0] * 10  # 被舔舐次数
        self.suckTimes = [0] * 10  # 吸乳次数
        self.noUpdatingTime = [0] * 10  # 追索机制
        self.st = [0] * 10
        self.cnt = 0
        self.white_pixel_count = 900
        self.threshold_value = 100
        self.init_ok = False
        self.ifUpdated = [0] * (self.total_num + 1)

    def change_num(self, num):
        self.total_num = num

    # def init_buffer(self, track_ids, bboxes):
    #     # 初始化缓冲区
    #     if len(self.buffer) >= self.total_num:
    #         self.init_ok = True
    #         return
    #     if len(track_ids) > 0 and len(self.buffer) < self.total_num:
    #         for track_id in track_ids:
    #             if self.cnt < self.total_num and self.st[track_id - 1] != 1:
    #                 w = bboxes[track_id - 1][2]  # 第3个元素是宽度
    #                 h = bboxes[track_id - 1][3]  # 第4个元素是高度
    #
    #                 # 计算面积
    #                 area = w * h
    #                 if area < 6400:
    #                     continue
    #                 self.buffer[track_id] = bboxes[track_id - 1]
    #                 self.buffer_origin[track_id] = bboxes[track_id - 1]
    #                 self.cnt += 1
    #                 self.st[track_id - 1] = 1
    #                 self.suckTimes[track_id] = 0

    def init_buffer(self, track_ids, bboxes):
        # 初始化缓冲区
        if len(self.buffer) >= self.total_num:
            self.init_ok = True
            return
        if len(track_ids) > 0 and len(self.buffer) < self.total_num:
            for track_id in track_ids:
                if self.cnt < self.total_num and self.st[track_id - 1] != 1:
                    w = bboxes[track_id - 1][2]  # 第3个元素是宽度
                    h = bboxes[track_id - 1][3]  # 第4个元素是高度

                    # 计算面积
                    area = w * h
                    if area < 6400:
                        continue
                    self.buffer[self.cnt + 1] = bboxes[track_id - 1]
                    self.buffer_origin[self.cnt + 1] = bboxes[track_id - 1]
                    self.st[self.cnt] = 1
                    self.suckTimes[self.cnt + 1] = 0
                    self.cnt += 1

    def check_stand_status(self, lamb_id):
        """检查羊是否站立"""
        if not self.stand[lamb_id]:
            origin_x = int(self.buffer_origin[lamb_id][0])
            origin_y = int(self.buffer_origin[lamb_id][1])
            current_x = int(self.buffer[lamb_id][0])
            current_y = int(self.buffer[lamb_id][1])

            if ((origin_x - current_x) ** 2 + (origin_y - current_y) ** 2 > 50000):
                self.stand[lamb_id] = 1
                return True
        return False

    def calculate_intersection_area(self, bbox1, bbox2, img):
        center_x1, center_y1, w1, h1 = bbox1
        center_x2, center_y2, w2, h2 = bbox2
        min_x1 = int(center_x1 - w1 // 2)
        min_x2 = int(center_x2 - w2 // 2)
        min_y1 = int(center_y1 - h1 // 2)
        min_y2 = int(center_y2 - h2 // 2)

        max_x1 = int(center_x1 + w1 // 2)
        max_x2 = int(center_x2 + w2 // 2)
        max_y1 = int(center_y1 + h1 // 2)
        max_y2 = int(center_y2 + h2 // 2)

        # 计算相交区域的左上角点
        inter_left_top_x = max(min_x1, min_x2)
        inter_left_top_y = max(min_y1, min_y2)

        # 计算相交区域的右下角点
        inter_right_bottom_x = min(max_x1, max_x2)
        inter_right_bottom_y = min(max_y1, max_y2)

        if inter_right_bottom_x - inter_left_top_x > 0 and inter_right_bottom_y - inter_left_top_y > 0:
            return img[inter_left_top_y - 3: inter_right_bottom_y + 3, inter_left_top_x - 3: inter_right_bottom_x + 3]
        else:
            return None

    def refresh_ifUpdated(self):
        self.ifUpdated = [0] * (self.total_num + 1)

    def in_buffer(self, bbox):
        for i in range(1, self.total_num + 1):
            t = 0
            while t < 4:
                if abs(bbox[t] - self.buffer[i][t]) < 1:
                    t += 1
                else:
                    break
            if t == 4:
                return True
        return False

    def position_update(self, bboxes, frame):
        for i in range(1, self.total_num + 1):
            if (self.ifUpdated[i] == 0):
                flag = False
                for box in bboxes:
                    if flag:
                        break
                    # if box in self.buffer.values():  # 如果该bbox已被使用则跳过
                    #     print(f"{box} in buffer")
                    #     print(self.buffer)
                    #     continue
                    if self.in_buffer(box) or box[2] * box[3] < 5400 or box[2] * box[3] > 12600:
                        continue

                    # 取得buffer中的一个点的坐标
                    # print(f"{i}'s area:{w * h}") #输出bbox面积
                    # if w * h < 5400 or w * h > 12500:
                    #     continue
                    x, y, w, h = box
                    # 否则进行如下操作
                    buffer_x, buffer_y, buffer_w, buffer_h = self.buffer[i]
                    top_left_x = int(buffer_x - buffer_w // 2)
                    top_left_y = int(buffer_y - buffer_h // 2)
                    bottom_right_x = int(buffer_x + buffer_w // 2)
                    bottom_right_y = int(buffer_y + buffer_h // 2)

                    # limit = -20

                    # cv2.rectangle(frame, (top_left_x - limit, top_left_y - limit),
                    #               (bottom_right_x + limit, bottom_right_y + limit),
                    #               (0, 0, 255))  # 绘制小羊的吸收框

                    # if top_left_x - limit < x < bottom_right_x + limit and top_left_y - limit < y < bottom_right_y + limit:
                    #     # color = get_color(i)
                    #     cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), thickness=5)
                    #     # 如果此时小羊的坐标在吸收框内则吸收该坐标,并用此时小羊的bbox更新吸收框
                    #     self.buffer[i] = box
                    #     self.ifUpdated[i] = 1  # 标记该小羊已被更新
                    # cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), thickness=4)
                    # cv2.circle(frame, (int(buffer_x), int(buffer_y)), 25, (255, 0, 0), thickness=2)
                    if (x - buffer_x) ** 2 + (y - buffer_y) ** 2 < 625:
                        self.buffer[i] = box
                        self.noUpdatingTime[i] = 0
                        self.ifUpdated[i] = 1  # 标记该小羊已被更新
                        flag = True

    def entityIOU_update(self, bboxes, frame, frame_count, save=False):
        # 创建保存图像的文件夹
        if save:
            save_dir = "tracking_process_images"
            intersect_dir = os.path.join(save_dir, "intersection")
            gray_dir = os.path.join(save_dir, "grayscale")
            binary_dir = os.path.join(save_dir, "binary")
            frame_dir = os.path.join(save_dir, "frame")
            # 确保文件夹存在
            for dir_path in [intersect_dir, gray_dir, binary_dir, frame_dir]:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            frame_filename = f"frame\_{frame_count}.png"
            cv2.imwrite(os.path.join(frame_dir, frame_filename), frame)

        for i in range(1, self.total_num + 1):
            if (self.ifUpdated[i] == 0):
                flag = False
                for bbox in bboxes:
                    if flag:
                        break

                    if self.in_buffer(bbox) or bbox[2] * bbox[3] < 5400 or bbox[2] * bbox[3] > 11000:
                        continue

                    # if bbox in self.buffer.values() or bbox[2] * bbox[3] < 5400 or bbox[2] * bbox[3] > 11000:
                    #     continue

                    intersect_region = self.calculate_intersection_area(bbox, self.buffer[i], frame)

                    if intersect_region is None:
                        continue
                    # 生成时间戳，用于文件名
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    if save:
                        # 保存交叉区域图像
                        intersect_filename = f"intersect_target{i}_{frame_count}.jpg"
                        cv2.imwrite(os.path.join(intersect_dir, intersect_filename), intersect_region)

                    # 转换为灰度图
                    gray_image = cv2.cvtColor(intersect_region, cv2.COLOR_BGR2GRAY)

                    if save:
                        gray_filename = f"gray_target{i}_{frame_count}.jpg"
                        cv2.imwrite(os.path.join(gray_dir, gray_filename), gray_image)

                    # 二值化处理并保存
                    ret, binary_image = cv2.threshold(gray_image, self.threshold_value, 255, cv2.THRESH_BINARY)
                    if save:
                        binary_filename = f"binary_target{i}_{frame_count}.jpg"
                        cv2.imwrite(os.path.join(binary_dir, binary_filename), binary_image)

                    # 统计二值图像中白色像素的数量
                    t = np.sum(binary_image == 255)
                    if t > self.white_pixel_count:
                        # cv2.imwrite(
                        #     f"E:/@code/@GraduationProject/@GraduationProject/intersect/{frame_count}_{i}.jpg",
                        #     intersect_region)
                        # cv2.imwrite(
                        #     f"E:/@code/@GraduationProject/@GraduationProject/binary/{frame_count}_{i}.jpg",
                        #     binary_image)
                        self.buffer[i] = bbox
                        self.noUpdatingTime[i] = 0
                        self.ifUpdated[i] = 1
                        flag = True
                if not flag:  # 遍历完所有的bbox flag仍为False则在该帧中没有被成功更新
                    self.noUpdatingTime[i] += 1

    def reID(self, bboxes):
        print(self.noUpdatingTime)
        for i in range(1, self.total_num + 1):
            if self.ifUpdated[i] == 0 and self.noUpdatingTime[i] > 2500:
                most_close = None
                noUpdate = self.buffer[i]
                for box in bboxes:
                    if self.in_buffer(box) or box[2] * box[3] > 10000 or box[2] * box[3] < 5400:
                        continue

                    # if box in self.buffer.values() or box[2] * box[3] > 10000 or box[2] * box[3] < 5400:
                    #     continue

                    usable = True

                    x, y, w, h = box
                    for m in range(1, self.total_num + 1):
                        x_t, y_t = self.buffer[m][0], self.buffer[m][1]
                        if (x - x_t) ** 2 + (y - y_t) ** 2 < 1000:
                            usable = False
                            print(f"{box} in {m}, can't be used")
                            break

                    if not usable:
                        continue

                    if most_close == None:
                        most_close = box
                    else:
                        x_no, y_no, w_no, h_no = noUpdate
                        x_m, y_m, w_m, h_m = most_close
                        x_c, y_c, w_c, h_c = box
                        distance_m = (x_no - x_m) ** 2 + (y_no - y_m) ** 2
                        distance_cur = (x_no - x_c) ** 2 + (y_no - y_c) ** 2
                        if i == 2:
                            print("id = 2的小羊正在使用id追索")
                        if distance_cur < distance_m:
                            most_close = box
                if most_close != None:
                    print(f"id = {i}的小羊正在使用id追索")
                    self.buffer[i] = most_close  # 使用找到的most_close去更新buffer
                    self.ifUpdated[i] = 1
                    self.noUpdatingTime[i] = 0  # 未更新次数清零

    def draw_boxes(self, image):
        """
        在图片上绘制所有边界框和对应的ID

        参数:
            image: numpy array, 输入图片
            buffer: 存储bbox信息的buffer字典或列表

        返回:
            draw_img: 绘制了边界框的图片
        """

        # 复制图片以免修改原图
        draw_img = image.copy()

        # 设置文字属性
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # 为不同ID设置不同的颜色
        def get_color(idx):
            # 使用与原代码相同的颜色生成方法
            cmap = cm.get_cmap('viridis', 7)
            color = cmap(idx)
            return tuple(int(255 * x) for x in color[:3])

        # 遍历buffer中的每个bbox
        for idx, bbox in self.buffer.items():
            if bbox is None:  # 跳过空的bbox
                continue
            # print(bbox)
            x, y, w, h = map(int, bbox)

            # 计算左上角和右下角坐标
            x1 = int(x - w // 2)
            y1 = int(y - h // 2)
            x2 = int(x + w // 2)
            y2 = int(y + h // 2)

            # 获取当前ID的颜色
            color = get_color(idx)

            # 绘制边界框
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)

            # 准备ID文本
            id_text = f'ID: {idx}'
            # if stand[idx]:
            #     state_text = 'Stand'
            # else:
            #     state_text = 'Prone'
            # display_text = f'{id_text} ({state_text})'

            font_params = {
                'font': cv2.FONT_HERSHEY_SIMPLEX,
                'font_scale': 0.8,
                'thickness': 2,
                'text_color': (255, 255, 255),  # 白色文字
                'bg_color': (0, 0, 0)  # 黑色背景
            }

            # 获取文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                id_text,
                font_params['font'],
                font_params['font_scale'],
                font_params['thickness']
            )

            # 绘制文本背景
            cv2.rectangle(draw_img,
                          (x1, y1 - text_height - 5),
                          (x1 + text_width, y1),
                          color,
                          -1)  # -1表示填充矩形

            # 绘制ID文本
            cv2.putText(draw_img,
                        id_text,
                        (x1, y1 - 5),
                        font,
                        font_scale,
                        (255, 255, 255),  # 白色文字
                        thickness)

        return draw_img


def main():
    root = tk.Tk()
    gui = GUI(root, "羊群监控系统")
    video_proc = VideoProcessor('./SRT_data/004.mp4',
                                './outputs/final1.mp4',
                                './outputs/final2.mp4')
    lamb_tracker = LambTracker()
    model = YOLO('lambBody-version2.pt')

    while video_proc.cap.isOpened():
        success, frame = video_proc.read_frame()

        while gui.is_paused():
            gui.update_self()

        if success:
            # 处理当前帧
            process_frame(frame, model, lamb_tracker, gui, video_proc)
        else:
            print('视频播放完毕')
            break

    root.mainloop()
    video_proc.release()
    cv2.destroyAllWindows()


def process_frame(frame, model, lamb_tracker, gui, video_proc):
    """处理单帧图像"""
    origin_frame = np.copy(frame)
    results = model.track(frame, persist=True)

    # 获取检测结果
    bboxes = results[0].boxes.xywh.cpu().tolist()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    annotated_frame = results[0].plot()

    # 初始化和更新追踪
    lamb_tracker.init_buffer(track_ids, bboxes, gui)
    if len(lamb_tracker.buffer) >= 5:
        update_tracking(frame, bboxes, lamb_tracker, gui, origin_frame)

    # 更新显示
    gui.update(annotated_frame, frame)
    video_proc.write_frames(frame, annotated_frame)


if __name__ == '__main__':
    main()
