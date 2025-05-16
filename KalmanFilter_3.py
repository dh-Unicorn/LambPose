import numpy as np


class KalmanFilter:
    def __init__(self, dt=3):
        """
        初始化卡尔曼滤波器
        :param dt: 时间步长
        """
        # 状态向量 [x, y, vx, vy]
        # x, y 是位置，vx, vy 是速度
        self.state = np.zeros(4)

        # 状态转移矩阵
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 观测矩阵
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 过程噪声协方差
        # self.Q = np.array([
        #     [0.1, 0, 0, 0],
        #     [0, 0.1, 0, 0],
        #     [0, 0, 0.2, 0],
        #     [0, 0, 0, 0.2]
        # ])
        self.Q = np.array([
            [0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 0.8, 0],  # 从0.2增加到0.8
            [0, 0, 0, 0.8]  # 从0.2增加到0.8
        ])
        # 测量噪声协方差
        # self.R = np.array([
        #     [0.1, 0],
        #     [0, 0.1]
        # ])
        self.R = np.array([
            [0.05, 0],    # 从0.1减小到0.05
            [0, 0.05]     # 从0.1减小到0.05
        ])

        # 状态协方差矩阵
        self.P = np.eye(4) * 1000

        # 是否已初始化
        self.initialized = False

    def init_state(self, x, y):
        """
        使用第一个观测值初始化状态
        """
        self.state = np.array([x, y, 0, 0])
        self.initialized = True

    def predict(self):
        """
        预测下一个状态
        """
        # 预测状态
        self.state = np.dot(self.A, self.state)
        # 预测误差协方差
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        # 返回预测的位置
        return self.state[0], self.state[1]

    def update(self, x, y):
        """
        使用观测值更新状态
        :param x: x坐标
        :param y: y坐标
        """
        if not self.initialized:
            self.init_state(x, y)
            return x, y

        measurement = np.array([x, y])

        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态
        y = measurement - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)

        # 更新误差协方差
        I = np.eye(4)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        return self.state[0], self.state[1]

    def get_next_position(self, x, y):
        """
        更新当前观测并预测下一个位置
        :param x: 当前观测的x坐标
        :param y: 当前观测的y坐标
        :return: 预测的下一个位置(x, y)
        """
        # 先用观测值更新状态
        self.update(x, y)
        # 然后预测下一个位置
        return self.predict()
