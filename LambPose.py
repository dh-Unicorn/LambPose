import sys
import cv2
import cvzone
import threading
import numpy as np
import torch
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from ultralytics import YOLO
from VideoReader import VideoReader
from LambTracker import LambTracker
from LambSkeleton import ProcessV2
from ewe_keypoints import eweKPdetector
from PredictBodyDirection import DirectionClassifier
from GUI.main import LambPoseWindow
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QPushButton, QTableWidget, QTableWidgetItem,
                             QHBoxLayout, QSplitter)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QTextEdit, QScrollBar, QMessageBox
from PyQt5.QtCore import QDateTime
from datetime import datetime
import os
import json
import sqlite3
from PyQt5.QtCore import QThread, pyqtSignal


# 保留所有原有的辅助函数
def ShowMidProcess(mid_process_imgs):
    for idx, mid_process_img in enumerate(mid_process_imgs):
        cv2.imshow("mid process" + str(idx), mid_process_img)


def get_color(i, total_colors=7):
    color_list = list(int((255 * i / total_colors + 50) % 255) for x in range(2))
    color_list.append(0)
    rgb_color = tuple(color_list)  # 将RGBA转换为RGB
    return rgb_color


def draw_mouseKP(frame, keypoints, color=(255, 0, 0), radius=8):
    x, y = int(keypoints[2][0]), int(keypoints[2][1])
    cv2.circle(frame, (x, y), radius, color, -1)  # 绘制关键点
    return [x, y]


def draw_eweKP(frame, results, color=(0, 255, 0), radius=5):
    """
    在图像上绘制 YOLO 检测到的关键点
    :param frame: 输入图像
    :param results: YOLO 检测结果
    :param color: 绘制关键点的颜色
    :param radius: 关键点的半径大小
    """
    eweMouseP = []
    mid_tail = []
    for result in results:  # 遍历每个检测结果
        if result is None:
            break
        keypoints = result.keypoints.xy.cpu().tolist()
        keypoints = keypoints[0]

        if keypoints is not None:
            cnt = 0
            for kp in keypoints:  # 遍历每个关键点
                if cnt == 2:
                    eweMouseP = draw_mouseKP(frame=frame, keypoints=keypoints)
                if cnt == 4 or cnt == 5:
                    mid_tail.append(kp[0])
                    mid_tail.append(kp[1])
                x, y = int(kp[0]), int(kp[1])  # 获取关键点坐标
                cv2.circle(frame, (x, y), radius, color, -1)  # 绘制关键点
                cnt += 1
    return eweMouseP, mid_tail


def get_color(i):
    # 预定义高对比度的颜色列表 (BGR格式)
    colors = [
        (0, 0, 255),  # 红色
        (0, 255, 0),  # 绿色
        (255, 0, 0),  # 蓝色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 洋红
        (255, 255, 0),  # 青色
        (128, 0, 255),  # 紫色
        (0, 128, 255),  # 橙色
        (255, 128, 0),  # 天蓝色
        (128, 255, 0),  # 黄绿色
    ]
    return colors[i % len(colors)]


def draw_judgingCircle(frame, i, buffer_origin):
    x = int(buffer_origin[i][0])
    y = int(buffer_origin[i][1])
    color = get_color(i)
    if i == 2:
        cv2.circle(frame, (x, y), 223, color, 2)


def get_draw_info(frame, lamb_tracker: LambTracker):
    stand = lamb_tracker.stand
    lickedTimes = lamb_tracker.lickedTimes
    suckTimes = lamb_tracker.suckTimes
    buffer = lamb_tracker.buffer

    for i in range(1, lamb_tracker.total_num + 1):
        info = " "
        # if stand[i]:
        #     info += "stand "
        # else:
        #     info += "prone "
        # info += str(int(lickedTimes[i] / 25))
        # info += ' '
        # info += str(int(suckTimes[i] / 25))
        cvzone.putTextRect(frame, info,
                           (int(buffer[i][0] - buffer[i][2] // 2) + 60, int(buffer[i][1] - buffer[i][3] // 2 - 5)),
                           scale=1,
                           thickness=2)


def StandDetect(frame, lamb_tracker: LambTracker):
    buffer_origin = lamb_tracker.buffer_origin
    buffer = lamb_tracker.buffer
    for i in range(1, lamb_tracker.total_num + 1):
        if (lamb_tracker.stand[i] == 0):
            draw_judgingCircle(frame, i, buffer_origin)  # 如果没有站立则继续绘制小羊的站立判断范围
            origin_x = int(buffer_origin[i][0])
            origin_y = int(buffer_origin[i][1])
            x = int(buffer[i][0])
            y = int(buffer[i][1])
            if ((origin_x - x) ** 2 + (origin_y - y) ** 2 > 50000):
                lamb_tracker.stand[i] = 1


def LickDetect(frame, lamb_tracker: LambTracker, eweMouseP):
    buffer = lamb_tracker.buffer
    for i in range(1, lamb_tracker.total_num + 1):
        x = int(buffer[i][0])
        y = int(buffer[i][1])
        if len(eweMouseP) == 0:
            return
        ewePx, ewePy = eweMouseP
        if (ewePx - x) ** 2 + (ewePy - y) ** 2 < 625:
            lamb_tracker.lickedTimes[i] += 1


def if_point_in_ellipse(px, py, cx, cy, a, b, angle):
    cos_angle = np.cos(np.radians(-angle))
    sin_angle = np.sin(np.radians(-angle))
    x = cos_angle * (px - cx) + sin_angle * (py - cy)
    y = -sin_angle * (px - cx) + cos_angle * (py - cy)
    return (x ** 2 / a ** 2) + (y ** 2 / b ** 2) <= 1


def SuckDetect(frame, lamb_tracker: LambTracker, mid_tail, mappedHeadP):
    if len(mid_tail) == 0:
        return
    midx, midy, tailx, taily = mid_tail

    # 计算椭圆参数
    center_x = (midx + tailx) / 2
    center_y = (midy + taily) / 2
    a = np.sqrt((tailx - midx) ** 2 + (taily - midy) ** 2) / 2 + 28
    if a > 100:
        return
    b = a / 2 + 15

    # 计算方向向量
    direction_x = tailx - midx
    direction_y = taily - midy
    direction_length = np.sqrt(direction_x ** 2 + direction_y ** 2)

    if direction_length > 0:
        unit_direction_x = direction_x / direction_length
        unit_direction_y = direction_y / direction_length
    else:
        unit_direction_x = 0
        unit_direction_y = 0

    shift_distance = a - 30
    new_center_x = center_x - unit_direction_x * shift_distance
    new_center_y = center_y - unit_direction_y * shift_distance
    angle = np.degrees(np.arctan2(taily - midy, tailx - midx))

    cv2.ellipse(frame,
                (int(new_center_x), int(new_center_y)),
                (int(a), int(b)),
                angle, 0, 360,
                (255, 0, 0), 2)

    for key, p in mappedHeadP.items():
        x = p[0]
        y = p[1]
        if if_point_in_ellipse(x, y, new_center_x, new_center_y, a, b, angle):
            lamb_tracker.suckTimes[key] += 1


class VideoProcessor:
    def __init__(self):
        self.init_models()
        self.init_video()
        self.init_trackers()
        self.frame_count = 0
        self.save = False
        self.pause = False
        self.use_lambTrack = True

        # 初始化视频保存器
        self.video_writer = None
        self.output_video_path = "whole_video.avi"  # 输出视频文件路径
        self.init_video_writer()

    def init_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.fps,  # 视频帧率
            (self.frame_width, self.frame_height)  # 视频分辨率
        )

    def init_models(self):
        self.model = YOLO('../weights/bestYOLOv9.pt')
        self.eweKPdetector = eweKPdetector.eweKPdetector()

        weightPath = "../weights/HeadPrediction.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.direction_model = DirectionClassifier(input_size=128)
        self.direction_model.load_state_dict(torch.load(weightPath))
        self.direction_model.to(self.device)

    def init_video(self):
        self.video_path = '../data/Whole.mp4'
        self.video_reader = VideoReader(self.video_path)

        self.frame_width = int(self.video_reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video_reader.cap.get(cv2.CAP_PROP_FPS))

    def init_trackers(self):
        self.lamb_tracker = LambTracker()
        self.lamb_tracker.change_num(4)

    def process_frame(self, frame):
        if self.pause:
            return None

        results = self.model.track(frame, conf=0.6)
        frame_byte = results[0].plot()

        bboxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        if not self.lamb_tracker.init_ok:
            self.lamb_tracker.init_buffer(track_ids, bboxes)
            return None
        # print("初始化完毕")
        back_up1 = frame.copy()
        back_up2 = frame.copy()
        self.lamb_tracker.refresh_ifUpdated()
        self.lamb_tracker.position_update(bboxes, frame)
        self.lamb_tracker.entityIOU_update(bboxes, back_up1, frame_count=self.frame_count)
        self.lamb_tracker.reID(bboxes)  # ID追索

        eweKPresults = self.eweKPdetector.detect(frame=back_up2)

        if len(eweKPresults) != 0:
            eweMouseP, mid_tail = draw_eweKP(frame=frame, results=eweKPresults)
            LickDetect(frame, self.lamb_tracker, eweMouseP)

        StandDetect(frame, self.lamb_tracker)
        _, mappedHeadP = ProcessV2(back_up2, frame, results, self.direction_model,
                                   self.lamb_tracker, self.device)

        SuckDetect(frame, self.lamb_tracker, mid_tail, mappedHeadP)
        get_draw_info(frame, self.lamb_tracker)

        annotated_img = self.lamb_tracker.draw_boxes(frame)
        # 保存处理后的帧到视频
        if self.video_writer is not None:
            self.video_writer.write(annotated_img)

        self.frame_count += 1
        if self.frame_count % 25 == 0:
            cv2.imwrite(f"E:/@code/@GraduationProject/@GraduationProject/wholeVideo/frame{self.frame_count}.jpg",
                        annotated_img)  # 保存所有帧

        cv2.imwrite(
            f"E:/@code/@GraduationProject/@GraduationProject/wholeVideo/{self.frame_count}{self.lamb_tracker.noUpdatingTime[:5]}.jpg",
            annotated_img)  # 保存所有帧
        # cv2.imwrite(
        #     f"E:/@code/@GraduationProject/@GraduationProject/wholeVideo_ORI/{self.frame_count}{self.lamb_tracker.noUpdatingTime[:5]}.jpg",
        #     frame_byte)

        stack = [annotated_img, frame_byte]
        return stack

    def release_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()


class DatabaseManager:
    def __init__(self):
        # 创建数据库连接
        self.conn = sqlite3.connect('lamb_monitoring.db')
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        """创建所需的数据表"""

        # 栏舍信息表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS pens (
            pen_id VARCHAR(10) PRIMARY KEY,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # 羔羊信息表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS lambs (
            lamb_id INTEGER,
            pen_id VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (lamb_id, pen_id),
            FOREIGN KEY (pen_id) REFERENCES pens(pen_id)
        )
        ''')

        # 事件类型表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS event_types (
            event_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name VARCHAR(50) UNIQUE,
            description TEXT
        )
        ''')

        # 事件日志表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS event_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP,
            pen_id VARCHAR(10),
            lamb_id INTEGER,
            event_type_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pen_id) REFERENCES pens(pen_id),
            FOREIGN KEY (event_type_id) REFERENCES event_types(event_type_id)
        )
        ''')

        # 初始化事件类型
        self.init_event_types()

        self.conn.commit()

    def init_event_types(self):
        """初始化事件类型数据"""
        event_types = [
            ('站立', '羔羊从卧下状态变为站立状态'),
            ('卧下', '羔羊从站立状态变为卧下状态'),
            ('被舔舐', '羔羊被母羊舔舐'),
            ('吸乳', '羔羊进行吸乳行为')
        ]

        self.cursor.executemany('''
        INSERT OR IGNORE INTO event_types (event_name, description)
        VALUES (?, ?)
        ''', event_types)

        self.conn.commit()

    def import_json_logs(self, json_file_path):
        """导入JSON格式的日志文件到数据库"""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        # 确保栏舍信息存在
        for log in logs:
            print("log:", log)
            pen_id = log['pen_id']
            self.cursor.execute('''
            INSERT OR IGNORE INTO pens (pen_id)
            VALUES (?)
            ''', (pen_id,))

            # 提取羔羊ID的数字部分
            lamb_id = int(log['lamb_id'].replace('小羊', ''))

            # 确保羔羊信息存在
            self.cursor.execute('''
            INSERT OR IGNORE INTO lambs (lamb_id, pen_id)
            VALUES (?, ?)
            ''', (lamb_id, pen_id))

            # 获取事件类型ID
            self.cursor.execute('''
            SELECT event_type_id FROM event_types
            WHERE event_name = ?
            ''', (log['action'],))
            event_type_id = self.cursor.fetchone()[0]

            # 插入事件日志
            self.cursor.execute('''
            INSERT INTO event_logs (timestamp, pen_id, lamb_id, event_type_id)
            VALUES (?, ?, ?, ?)
            ''', (
                log['timestamp'],
                pen_id,
                lamb_id,
                event_type_id
            ))

        self.conn.commit()

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


class DatabaseWorker(QThread):
    """数据库操作工作线程"""
    error_occurred = pyqtSignal(str)  # 错误信号

    def __init__(self):
        super().__init__()
        self.db_manager = None
        self.queue = []
        self.running = True

    def run(self):
        """线程运行函数"""
        self.db_manager = DatabaseManager()
        while self.running:
            if self.queue:
                data = self.queue.pop(0)
                self.process_data(data)
            else:
                self.msleep(5)  # 短暂休眠避免CPU占用过高

    def process_data(self, data):
        """处理数据库操作"""
        try:
            timestamp, pen_id, lamb_id, action = data

            # 确保栏舍信息存在
            self.db_manager.cursor.execute('''
            INSERT OR IGNORE INTO pens (pen_id)
            VALUES (?)
            ''', (pen_id,))

            # 确保羔羊信息存在
            self.db_manager.cursor.execute('''
            INSERT OR IGNORE INTO lambs (lamb_id, pen_id)
            VALUES (?, ?)
            ''', (lamb_id, pen_id))

            # 获取事件类型ID
            self.db_manager.cursor.execute('''
            SELECT event_type_id FROM event_types
            WHERE event_name = ?
            ''', (action,))
            event_type_id = self.db_manager.cursor.fetchone()[0]

            # 插入事件日志
            self.db_manager.cursor.execute('''
            INSERT INTO event_logs (timestamp, pen_id, lamb_id, event_type_id)
            VALUES (?, ?, ?, ?)
            ''', (timestamp, pen_id, lamb_id, event_type_id))

            # 提交事务
            self.db_manager.conn.commit()

        except sqlite3.Error as e:
            self.error_occurred.emit(str(e))

    def add_task(self, data):
        """添加数据库任务"""
        self.queue.append(data)

    def stop(self):
        """停止线程"""
        self.running = False
        if self.db_manager:
            self.db_manager.close()


class EnhancedLambPoseWindow(LambPoseWindow):
    def __init__(self):
        super().__init__()
        self.processor = VideoProcessor()
        self.current_frame = None
        self.previous_states = {}
        self.previous_licks = {}
        self.previous_sucks = {}
        self.pauseFrame0 = None
        self.pauseFrame1 = None
        # 创建数据库工作线程
        self.db_worker = DatabaseWorker()
        self.db_worker.error_occurred.connect(self.handle_db_error)
        self.db_worker.start()

        # 初始化日志相关属性
        self.pen_id = "01"  # 栏舍编号
        self.event_logs = []  # 用于保存到JSON文件的日志列表
        self.log_dir = Path("event_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 初始化状态
        for i in range(1, 6):
            self.previous_states[i] = False
            self.previous_licks[i] = 0
            self.previous_sucks[i] = 0

        self.save_dir = Path("saved_frames")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.start_processing()

    def handle_db_error(self, error_msg):
        """处理数据库错误"""
        print(f"数据库错误: {error_msg}")
        QMessageBox.warning(self, "数据库错误", f"记录事件时发生错误: {error_msg}")

    def log_to_database(self, timestamp, lamb_id, action):
        """直接将事件记录到数据库"""
        try:
            # 确保栏舍信息存在
            self.db_manager.cursor.execute('''
            INSERT OR IGNORE INTO pens (pen_id)
            VALUES (?)
            ''', (self.pen_id,))

            # 确保羔羊信息存在
            self.db_manager.cursor.execute('''
            INSERT OR IGNORE INTO lambs (lamb_id, pen_id)
            VALUES (?, ?)
            ''', (lamb_id, self.pen_id))

            # 获取事件类型ID
            self.db_manager.cursor.execute('''
            SELECT event_type_id FROM event_types
            WHERE event_name = ?
            ''', (action,))
            event_type_id = self.db_manager.cursor.fetchone()[0]

            # 插入事件日志
            self.db_manager.cursor.execute('''
            INSERT INTO event_logs (timestamp, pen_id, lamb_id, event_type_id)
            VALUES (?, ?, ?, ?)
            ''', (timestamp, self.pen_id, lamb_id, event_type_id))

            # 提交事务
            self.db_manager.conn.commit()

        except sqlite3.Error as e:
            print(f"数据库错误: {e}")
            QMessageBox.warning(self, "数据库错误", f"记录事件时发生错误: {e}")

    def add_log_entry(self, lamb_id, action):
        """添加日志条目到JSON文件并同时写入数据库"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 创建日志条目
        log_entry = {
            "timestamp": timestamp,
            "pen_id": self.pen_id,
            "lamb_id": f"小羊{lamb_id}",
            "action": action
        }

        # 添加到JSON日志列表
        self.event_logs.append(log_entry)

        # 将数据添加到数据库工作线程的队列
        self.db_worker.add_task((timestamp, self.pen_id, lamb_id, action))

        # 更新界面日志显示
        log_text = f"{timestamp} - 小羊{lamb_id}: {action}"
        self.log_text.append(log_text)

        # 定期保存JSON文件
        self.save_json_logs()

    def save_json_logs(self):
        """保存日志到JSON文件"""
        if not self.event_logs:
            return

        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"{current_date}_pen{self.pen_id}.json"
        file_path = self.log_dir / filename

        # 保存到JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.event_logs, f, ensure_ascii=False, indent=2)

    def save_event_logs(self):
        """保存事件日志到文件和数据库"""
        if not self.event_logs:
            return

        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"{current_date}_pen{self.pen_id}.json"
        file_path = self.log_dir / filename

        # 保存到JSON文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.event_logs, f, ensure_ascii=False, indent=2)

        # 导入到数据库
        self.db_manager.import_json_logs(file_path)

        # 清空当前日志缓存
        self.event_logs = []

    def on_close(self, event):
        """窗口关闭时的处理"""
        # 最后一次保存JSON文件
        self.save_json_logs()
        # 停止数据库工作线程
        self.db_worker.stop()
        self.db_worker.wait()
        event.accept()

    def set_pen_id(self, pen_id):
        """设置栏舍编号"""
        self.pen_id = pen_id

    def check_and_log_events(self, lamb_id, standing, licks, sucks):
        """检查状态变化并记录事件"""
        # 检查站立状态变化
        if standing != self.previous_states[lamb_id]:
            action = "站立" if standing else "卧下"
            self.add_log_entry(lamb_id, action)
            self.previous_states[lamb_id] = standing

        # 检查舔舐次数变化
        if licks > self.previous_licks[lamb_id]:
            self.add_log_entry(lamb_id, "被舔舐")
            self.previous_licks[lamb_id] = licks

        # 检查吸乳次数变化
        if sucks > self.previous_sucks[lamb_id]:
            self.add_log_entry(lamb_id, "吸乳")
            self.previous_sucks[lamb_id] = sucks

    def start_processing(self):
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def update_lamb_info(self, row, lamb_id, standing, licks, suckles):
        # 在主线程中更新UI
        self.lamb_info_table.setRowCount(max(self.lamb_info_table.rowCount(), row + 1))

        # 创建新的表格项并设置
        id_item = QTableWidgetItem(lamb_id)
        standing_item = QTableWidgetItem(standing)
        licks_item = QTableWidgetItem(str(licks))
        suckles_item = QTableWidgetItem(str(suckles))

        # 设置字体
        font = QFont("Times New Roman", 12)
        for item in [id_item, standing_item, licks_item, suckles_item]:
            item.setFont(font)

        # 设置项目到表格
        self.lamb_info_table.setItem(row, 0, id_item)
        self.lamb_info_table.setItem(row, 1, standing_item)
        self.lamb_info_table.setItem(row, 2, licks_item)
        self.lamb_info_table.setItem(row, 3, suckles_item)

        # 强制更新表格
        self.lamb_info_table.viewport().update()

    def process_video(self):
        for frame in self.processor.video_reader.read_frames():
            imgStack = self.processor.process_frame(frame)

            if imgStack is not None:
                if not self.processor.pause:
                    self.pauseFrame0 = imgStack[0]
                    self.pauseFrame1 = imgStack[1]
                if self.use_LambTrack:
                    if self.processor.pause:
                        self.update_frame(self.pauseFrame0)
                    self.current_frame = imgStack[0]
                    self.update_frame(imgStack[0])
                else:
                    if self.processor.pause:
                        self.update_frame(self.pauseFrame1)
                    self.current_frame = imgStack[1]
                    self.update_frame(imgStack[1])

                # 更新小羊信息表并检查事件
                for i in range(1, self.processor.lamb_tracker.total_num + 1):
                    standing = bool(self.processor.lamb_tracker.stand[i])
                    licks = int(self.processor.lamb_tracker.lickedTimes[i] / 25)
                    sucks = int(self.processor.lamb_tracker.suckTimes[i] / 25)

                    self.update_lamb_info(
                        i - 1,
                        str(i),
                        "是" if standing else "否",
                        licks,
                        sucks
                    )

                    # 检查并记录事件
                    self.check_and_log_events(i, standing, licks, sucks)

    def pause_video(self):
        self.processor.pause = True

    def continue_video(self):
        self.processor.pause = False

    def save_frame(self):
        """保存当前帧到本地"""
        if self.current_frame is not None:
            # 生成文件名（使用时间戳）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = self.save_dir / f"frame_{timestamp}.jpg"

            # 保存图片
            cv2.imwrite(str(filename), self.current_frame)

            # 显示保存成功的消息
            QMessageBox.information(self, "保存成功", f"当前帧已保存至：\n{filename}")
        else:
            QMessageBox.warning(self, "保存失败", "当前没有可保存的帧")

    def switch_model(self):
        if self.use_LambTrack:
            self.use_LambTrack = False
        else:
            self.use_LambTrack = True


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EnhancedLambPoseWindow()
    window.show()
    sys.exit(app.exec_())
    window.processor.release_video_writer()
