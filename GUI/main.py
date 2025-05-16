import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QWidget, QPushButton, QTableWidget, QTableWidgetItem,
                             QHBoxLayout, QSplitter)
from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap, QFont
import cv2
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QTextEdit, QScrollBar, QMessageBox
from PyQt5.QtGui import QColor


class LambPoseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stop = False
        self.initUI()
        self.use_LambTrack = True

    def initUI(self):
        self.setWindowTitle('Video Display')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
                border-radius: 10px;
            }
        """)

        # 创建中央窗口部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 创建左侧布局
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # 创建视频显示标签
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        # self.video_label.setMinimumSize(1425, 900)
        self.video_label.setFixedSize(1425, 900)  # 固定大小
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)

        # 创建按钮布局
        button_layout = QHBoxLayout()

        # 按钮字体
        button_font = QFont("宋体", 14)

        # 创建暂停按钮
        self.pause_button = QPushButton('暂停', self)
        self.pause_button.setFont(button_font)
        self.pause_button.setStyleSheet(self.button_style())
        self.pause_button.clicked.connect(self.pause_video)

        # 创建继续按钮
        self.continue_button = QPushButton('继续', self)
        self.continue_button.setFont(button_font)
        self.continue_button.setStyleSheet(self.button_style())
        self.continue_button.clicked.connect(self.continue_video)

        # 创建保存按钮
        self.save_frame_button = QPushButton('保存该帧', self)
        self.save_frame_button.setFont(button_font)
        self.save_frame_button.setStyleSheet(self.button_style())
        self.save_frame_button.clicked.connect(self.save_frame)

        # 创建切换模型按钮
        self.switch_model_button = QPushButton('切换模型', self)
        self.switch_model_button.setFont(button_font)
        self.switch_model_button.setStyleSheet(self.button_style())
        self.switch_model_button.clicked.connect(self.switch_model)

        # 将按钮添加到布局
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.continue_button)
        button_layout.addWidget(self.save_frame_button)
        button_layout.addWidget(self.switch_model_button)

        # 将视频标签和按钮布局添加到左侧布局
        left_layout.addWidget(self.video_label)
        left_layout.addLayout(button_layout)

        # 创建右侧布局
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # 创建兰舍信息表格
        self.barn_info_table = QTableWidget(6, 2, self)
        self.barn_info_table.verticalHeader().setVisible(False)
        self.barn_info_table.horizontalHeader().setVisible(False)
        self.barn_info_table.setItem(0, 0, QTableWidgetItem('当前时间'))
        self.barn_info_table.setItem(1, 0, QTableWidgetItem('栏舍编号'))
        self.barn_info_table.setItem(2, 0, QTableWidgetItem('母羊 ID'))
        self.barn_info_table.setItem(3, 0, QTableWidgetItem('温度'))
        self.barn_info_table.setItem(4, 0, QTableWidgetItem('湿度'))
        self.barn_info_table.setItem(5, 0, QTableWidgetItem('监控状态'))
        self.barn_info_table.setStyleSheet(self.table_style())

        # 设置字体
        font_english = QFont("Times New Roman", 12)
        font_chinese = QFont("宋体", 12)

        for i in range(6):
            self.barn_info_table.item(i, 0).setFont(font_chinese)
            self.barn_info_table.setItem(i, 1, QTableWidgetItem("示例信息"))
            self.barn_info_table.item(i, 1).setFont(font_english)

        self.barn_info_table.setItem(1, 1, QTableWidgetItem("1"))
        self.barn_info_table.setItem(2, 1, QTableWidgetItem("1"))
        self.barn_info_table.setItem(3, 1, QTableWidgetItem("26℃"))
        self.barn_info_table.setItem(4, 1, QTableWidgetItem("20%"))

        item = QTableWidgetItem("正常")
        item.setForeground(QColor(0, 180, 0))  # 深一点的绿色
        # 或者
        item.setForeground(QColor(34, 139, 34))  # ForestGreen
        # 或者
        item.setForeground(QColor(60, 179, 113))  # MediumSeaGreen
        self.barn_info_table.setItem(5, 1, item)

        # 调整栏舍信息第二列宽度
        self.barn_info_table.setColumnWidth(1, 150)
        self.barn_info_table.horizontalHeader().setStretchLastSection(True)

        # 创建小羊状态表格
        self.lamb_info_table = QTableWidget(1, 4, self)
        self.lamb_info_table.setHorizontalHeaderLabels(['ID', '是否站立', '被舔舐次数', '吸乳次数'])
        self.lamb_info_table.verticalHeader().setVisible(False)
        self.lamb_info_table.setStyleSheet(self.table_style())
        self.lamb_info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        # 设置字体
        for i in range(4):
            self.lamb_info_table.horizontalHeaderItem(i).setFont(font_english)

        # 调整列宽
        self.lamb_info_table.setColumnWidth(0, 50)  # ID列适当窄一些
        self.lamb_info_table.horizontalHeader().setStretchLastSection(True)

        # 创建文本显示区域
        self.log_text = QTextEdit(self)  # 添加 QTextEdit 的导入
        self.log_text.setReadOnly(True)  # 设置为只读
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #d0d0d0;
                padding: 10px;
                font-family: "Times New Roman";
                font-size: 12pt;
            }
        """)

        # 设置标题字体
        title_font = QFont("宋体", 14, QFont.Bold)
        # 将组件添加到右侧布局
        barn_info_label = QLabel("栏舍信息", self)
        barn_info_label.setFont(title_font)
        right_layout.addWidget(barn_info_label)
        right_layout.addWidget(self.barn_info_table, 1)

        lamb_info_label = QLabel("羔羊状态", self)
        lamb_info_label.setFont(title_font)
        right_layout.addWidget(lamb_info_label)
        right_layout.addWidget(self.lamb_info_table, 2)

        image_info_label = QLabel("事件日志", self)
        image_info_label.setFont(title_font)
        right_layout.addWidget(image_info_label)
        right_layout.addWidget(self.log_text, 1)

        # 使用 QSplitter 将左右布局分隔
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        # splitter.setHandleWidth(0)
        splitter.setStretchFactor(0, 0.5)  # 左侧分配更多空间
        splitter.setStretchFactor(1, 1)  # 右侧分配较少空间

        # 将分隔器添加到主布局
        main_layout.addWidget(splitter)

        # 创建定时器更新时间
        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)

        self.showMaximized()

    def button_style(self):
        return """
            QPushButton {
                background-color: #007aff;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
        """

    def table_style(self):
        return """
            QTableWidget {
                border-radius: 10px;
                border: 1px solid #e0e0e0;
                background-color: white;
            }
        """

    def update_frame(self, frame):
        if frame is not None:
            # 将 BGR 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(),
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)

    def update_time(self):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        self.barn_info_table.setItem(0, 1, QTableWidgetItem(current_time))
        self.barn_info_table.item(0, 1).setFont(QFont("Times New Roman", 12))

    def update_lamb_info(self, row, lamb_id, standing, licks, suckles):
        self.lamb_info_table.setRowCount(max(self.lamb_info_table.rowCount(), row + 1))
        self.lamb_info_table.setItem(row, 0, QTableWidgetItem(lamb_id))
        self.lamb_info_table.setItem(row, 1, QTableWidgetItem(standing))
        self.lamb_info_table.setItem(row, 2, QTableWidgetItem(str(licks)))
        self.lamb_info_table.setItem(row, 3, QTableWidgetItem(str(suckles)))

    def update_key_image(self, image_path):
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def pause_video(self):
        self.stop = True

    def continue_video(self):
        self.stop = False

    def save_frame(self):
        print("保存当前帧")

    def switch_model(self):
        print("切换模型")

    def add_log_entry(self, lamb_id, action):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        log_entry = f"{current_time} - 小羊{lamb_id}: {action}\n"
        self.log_text.append(log_entry)
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LambPoseWindow()
    sys.exit(app.exec_())
