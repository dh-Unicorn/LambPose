import cv2
import os


def extract_frames(video_path, output_dir):
    """
    从视频中按照每秒一帧的频率抽取帧，并保存到指定目录。

    :param video_path: 输入视频的路径
    :param output_dir: 保存帧的输出目录
    """
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频的帧率和总帧数
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频每秒的帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度

    print(f"视频信息: 分辨率={frame_width}x{frame_height}, 帧率={fps}, 总帧数={total_frames}")

    # 按照每秒一帧的频率抽取帧
    frame_interval = fps  # 每秒一帧
    frame_number = 1  # 保存帧的起始编号

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # 仅保存符合间隔的帧
        if frame_idx % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{frame_number}.jpg")
            cv2.imwrite(output_path, frame)  # 保存帧
            print(f"保存帧: {output_path}")
            frame_number += 1

    # 释放视频资源
    cap.release()
    print(f"帧提取完成，共保存 {frame_number - 1} 帧到目录: {output_dir}")


# 使用示例
if __name__ == "__main__":
    video_path = "E:/desktop/D05.mp4"  # 输入视频路径
    output_dir = "E:/desktop/t"  # 输出帧保存目录
    extract_frames(video_path, output_dir)
