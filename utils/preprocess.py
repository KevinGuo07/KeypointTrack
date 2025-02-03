import cv2
import numpy as np
import os


def load_image(filepath):
    """
    Args:
        filepath (str): 图像文件的路径。

    Returns:
        np.ndarray: 加载的 RGB 图像，形状为 (H, W, 3)。
    """
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法加载图像文件：{filepath}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_pointcloud(filepath):
    """
    Args:
        filepath (str): 点云文件的路径（.npy 格式）。

    Returns:
        np.ndarray: 点云数据，形状为 (H, W, 3)。
    """
    try:
        # 确认文件存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"点云文件不存在：{filepath}")

        # 加载点云文件
        points = np.load(filepath)
        # print(f"Loaded point cloud shape: {points.shape}, ndim: {points.ndim}")

        # 检查点云格式是否符合要求
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError(f"点云文件格式错误，期望 (H * W, 3)，实际 {points.shape}")

        return points
    except Exception as e:
        raise ValueError(f"无法加载点云文件：{filepath}, 错误：{e}")


def load_mask(filepath):
    """
    Args:
        filepath (str): 掩码文件的路径（.npy 格式）。

    Returns:
        np.ndarray: 掩码数据，形状为 (H, W)，整数值表示物体类别。
    """
    try:
        mask = np.load(filepath)
        if mask.ndim != 2:
            raise ValueError(f"掩码文件格式错误：{filepath}")
        mask_flattened = mask  # .flatten()
        return mask_flattened
    except Exception as e:
        raise ValueError(f"无法加载掩码文件：{filepath}, 错误：{e}")


def save_image(image, filepath):
    """
    Args:
        image (np.ndarray): 待保存的 RGB 图像，形状为 (H, W, 3)。
        filepath (str): 保存图像的路径。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(filepath, image_bgr)
    if not success:
        print(f"保存图片失败：{filepath}")
