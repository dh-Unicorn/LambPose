import cv2


class VideoReader:
    def __init__(self, video_path):
        """
        初始化视频读取器

        参数:
            video_path: 视频文件路径
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error: 无法打开视频文件 {video_path}")

        # 获取视频属性
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_frame(self):
        """读取一帧"""
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def read_frames(self):
        """生成器：逐帧读取视频"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def set_frame_position(self, frame_number):
        """设置当前帧位置"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_video_info(self):
        """获取视频信息"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.frame_count / self.fps
        }

    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release()
