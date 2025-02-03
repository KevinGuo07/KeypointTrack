import cv2
import os
from tqdm import tqdm


def create_video_from_images(image_files, output_video_path, frame_rate=2):
    # define valid extension
    if not image_files:
        raise ValueError("No valid image files found.")

        # 加载第一张图片以获取视频尺寸
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # 写入每一张图片
    for image_path in tqdm(image_files, desc="Writing video"):
        image = cv2.imread(image_path)
        video_writer.write(image)

    video_writer.release()
    print(f"Video saved at {output_video_path}")

