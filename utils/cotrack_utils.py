import json
import math
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from utils.keypoint_utils import KeypointProposer
from utils.preprocess import load_image, load_pointcloud, load_mask, save_image


def sort_files_by_number(directory, prefix):
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]

    def extract_number(filename):
        match = re.search(fr'{prefix}(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    return sorted(files, key=extract_number)


def get_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/config.yaml')
    assert os.path.exists(config_path), "Config file does not exist: {config_path}"
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def obj_dict_generate(all_json_save_path, base_path):
    """生成包含所有帧信息的字典"""
    if not os.path.exists(all_json_save_path):
        raise FileNotFoundError(f"JSON file not found: {all_json_save_path}")

    with open(all_json_save_path, 'r') as f:
        data = json.load(f)

    obs_dicts = {}
    for step_idx_str in tqdm(sorted(data.keys(), key=lambda x: int(x))):
        guid_data = data[step_idx_str]

        paths = {
            "rgb": os.path.join(base_path, guid_data["rgb_image_npy"]),
            "depth": os.path.join(base_path, guid_data["depth_image_npy"]),
            "masks": os.path.join(base_path, guid_data["masks_npy"]),
        }

        if not all(os.path.exists(p) for p in paths.values()):
            print(f"Files missing for step {step_idx_str}, skipping.")

            for key, path in paths.items():
                if not os.path.exists(path):
                    print(f"Missing {key}: {path}")
            continue

        obs_dicts[int(step_idx_str)] = {
            "rgb_image": np.load(paths["rgb"]),
            "depth_image": np.load(paths["depth"]),
            "masks": np.load(paths["masks"]),
            "info": guid_data["info"],
            "frame_id": step_idx_str
        }

    return obs_dicts


def save_and_show_keypoints(image, keypoint_info, image_path, output_dir, index):
    object_ids = sorted(set(info['object_id'] for info in keypoint_info))
    color_map = plt.colormaps['tab10']  # 获取 'tab10' colormap
    colors = [color_map(i / len(object_ids)) for i in range(len(object_ids))]  # 按比例截取所需颜色
    color_dict = {obj_id: colors[i] for i, obj_id in enumerate(object_ids)}

    plt.figure(figsize=(8, 6))
    plt.imshow(image)

    for info in keypoint_info:
        pixel_coords = info['pixel_coords']
        if pixel_coords is None:
            continue
        else:
            obj_id = info['object_id']

        # 根据 object_id 从 color_dict 中获取对应颜色
        color = color_dict[obj_id]
        plt.scatter(pixel_coords[0], pixel_coords[1], s=50, edgecolors='white', color=color,
                    label=f"bottle_id: {obj_id}")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=os.path.basename(image_path))
    plt.axis('off')

    output_path = os.path.join(output_dir, f"keypoints_{index}.jpg")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved keypoints image to: {output_path}")


def get_point_cloud(obs_dict, point_seq_dir):
    depth_image, info = obs_dict["depth_image"], obs_dict["info"]
    H, W = depth_image.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_matrix_t = torch.tensor(info["model_matrix"], dtype=torch.float32).to(device)

    y, x = np.indices((H, W))
    x, y, z = [torch.from_numpy(arr).float().to(device) for arr in [x, y, depth_image / 1000.0]]

    fov_rad = info["fov"] / 180.0 * math.pi
    focal_length = W / (2.0 * math.tan(fov_rad / 2.0))

    x_cam = (x - W / 2.0) * z / focal_length
    y_cam = (y - H / 2.0) * z / focal_length
    points_cam = torch.stack((x_cam, y_cam, z), dim=2).view(-1, 3).unsqueeze(0)

    rotation = model_matrix_t[:3, :3].transpose(0, 1).unsqueeze(0)
    translation = model_matrix_t[:3, 3].unsqueeze(0)
    points_world = torch.bmm(points_cam, rotation) + translation

    points_world_np = points_world.squeeze(0).cpu().numpy().reshape(H, W, 3)
    os.makedirs(point_seq_dir, exist_ok=True)
    np.save(os.path.join(point_seq_dir, f"point_{obs_dict['frame_id']}.npy"), points_world_np.astype(np.float16))


def generate_point_clouds(obj_dicts, point_seq_dir):
    """为所有帧生成点云"""
    for frame_id, obs_dict in obj_dicts.items():
        try:
            get_point_cloud(obs_dict, point_seq_dir)
        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")


def extract_tracked_keypoints(i, keypoints_batch, keypoint_info_batch, model_step, if_firstbatch=0):
    """
    从 keypoints_batch 和 keypoint_info_batch 中提取 [i-model_step, i) 帧的关键点信息。

    参数：
        i (int): 当前帧的索引
        keypoints_batch (Tensor): 关键点张量batch
        keypoint_info_batch (list): 包含字典的列表
        model_step (int): 模型的步长

    返回：
        tracked_keypoints (list): 每个元素为 (frame_keypoints, frame_info)，
    """
    tracked_keypoints = []
    batch_step = model_step * if_firstbatch
    start_frame = i - model_step-batch_step
    for j in range(start_frame, i-batch_step):
        # 提取当前帧 j 的所有信息
        frame_info = [info for info in keypoint_info_batch if info.get("frame") == j]

        # 计算 keypoints_batch 中对应当前帧的索引
        frame_indices = [
            info["frame"] - start_frame
            for info in keypoint_info_batch
            if isinstance(info, dict)
            and "frame" in info
            and start_frame <= info["frame"] < i
            and (info["frame"] - start_frame) < keypoints_batch.shape[0]
        ]

        frame_keypoints = keypoints_batch[frame_indices] if frame_indices else None

        if frame_keypoints is not None:
            tracked_keypoints.append((frame_keypoints, frame_info))

    return tracked_keypoints


class KeypointTracker:
    def __init__(self, model=None):
        kp_config = get_config(config_path="kp_config.yaml")
        self.keypoint_proposer = KeypointProposer(kp_config['keypoint_proposer'])
        self.model = model
        if model is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def _process_step(self, window_frames, is_first_step, query, grid_query_frame):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-self.model.step * 2:]), device=self.device
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return self.model(
            video_chunk,
            is_first_step=is_first_step,
            grid_query_frame=grid_query_frame,
            queries=query
        )

    def keypoint_track(self, dirs, base_path, task_path):
        image_files = sort_files_by_number(dirs["image"], prefix="rgb_")
        mask_files = sort_files_by_number(dirs["mask"], prefix="actor_seg_")
        obj_dicts_sorted = obj_dict_generate(dirs["info"], base_path)
        generate_point_clouds(obj_dicts_sorted, dirs["pointcloud"])
        point_files = sort_files_by_number(dirs["pointcloud"], prefix="point_")

        assert len(image_files) == len(point_files) == len(mask_files), "序列文件数量不一致！"

        tracked_keypoints = []
        window_frames = []

        for i, (img_file, point_file, mask_file) in enumerate(zip(image_files, point_files, mask_files)):
            image = load_image(os.path.join(dirs["image"], img_file))
            points = load_pointcloud(os.path.join(dirs["pointcloud"], point_file))
            mask = load_mask(os.path.join(dirs["mask"], mask_file))

            H, W, _ = image.shape

            if i == 0:
                keypoints, keypoint_info, keypoints_points, features_flat = self.initialize_keypoint(image, points,
                                                                                                     mask,
                                                                                                     task_path)
                queries = self._initialize_queries(keypoints)
                tracked_keypoints.append((keypoints, keypoint_info))

            if i % self.model.step == 0 and i != 0:
                tracked_keypoints = self.cotrack_keypoints(image, keypoints, i, queries, tracked_keypoints,
                                                               window_frames, keypoint_info)
            window_frames.append(image)
        # if index of the last frame is not the multiple of step:
        image = load_image(os.path.join(dirs["image"], image_files[-1]))
        last_batch_start = -(i % self.model.step) - self.model.step - 1
        tracked_keypoints = self.cotrack_keypoints(image, keypoints, i, queries, tracked_keypoints,
                                                                      window_frames[last_batch_start:], keypoint_info)
        tracked_keypoints.pop(0)
        return tracked_keypoints

    def initialize_keypoint(self, image, points, mask, task_path=None):
        keypoints, projected_img, candidate_pixels, features_flat = self.keypoint_proposer.get_keypoints(image, points,
                                                                                                         mask)
        real_keypoints, keypoint_info = self.keypoint_proposer.register_keypoints(keypoints, points, mask)

        if task_path:
            save_image(projected_img, os.path.join(task_path, "projected_keypoints.jpg"))
        print("keypoint of first input image is initialized!")
        # print(candidate_pixels)
        # print("shape of image is:", image.shape, "shape of points is:", points.shape, "shape of mask is:", mask.shape)
        return candidate_pixels, keypoint_info, real_keypoints, features_flat

    # 此时可以拿到的输入有：上一帧的参考点坐标和对应的特征信息，当前帧的rgb、点云、mask
    # 考虑加入的输入：
    def cotrack_keypoints(self, image, keypoint, iteration, queries, tracked_keypoints, window_frames, keypoint_info):
        """
        Args:
            image (np.ndarray): 当前帧图像，形状为 (H, W, 3)。
            keypoint (np.ndarray): 初始关键点，形状为 (N, 3)，包含 x, y, z。
            iteration (int): 当前迭代次数，1 表示初始化帧。
            queries: 查询点索引的坐标
            window_frames: 目前的窗口帧集合
            keypoint_info: 包含keypoint"frame" "object_id"  "pixel_coords"的字典
        Returns:
            keypoints_tensor (torch.Tensor): 当前帧的关键点坐标，形状为 (N, 3)。
            keypoint_info (list): 包含关键点的位置信息的字典列表。
        """

        queries = queries.to(self.device)
        if iteration == self.model.step:
            tracks, vis = self._process_step(
                window_frames,
                is_first_step=True,
                query=queries,
                grid_query_frame=0,
            )
            window_frames.append(image)
            keypoints_tensor = self._process_tracks(tracks, vis, keypoint, keypoint_info)

        else:
            tracks, vis = self._process_step(
                window_frames,
                is_first_step=False,
                query=queries,
                grid_query_frame=0,  # 检查是否可以不要
            )
            # because the first "step" images are not recorded, add extra
            if iteration == self.model.step * 2:
                print(f"Frame {iteration}: tracks shape：{tracks.shape} with first batch tracked")
                window_frames.append(image)
                keypoints_tensor = self._process_tracks_firstbatch(tracks, vis, keypoint, keypoint_info)
                new_tracked_keypoints = extract_tracked_keypoints(iteration, keypoints_tensor,
                                                                  keypoint_info, self.model.step, if_firstbatch=True)
                tracked_keypoints.extend(new_tracked_keypoints)
                return tracked_keypoints

            print(f"Frame {iteration}: tracks shape：{tracks.shape}")
            window_frames.append(image)
            keypoints_tensor = self._process_tracks(tracks, vis, keypoint, keypoint_info)

        new_tracked_keypoints = extract_tracked_keypoints(iteration, keypoints_tensor,
                                                          keypoint_info, self.model.step, if_firstbatch=False)
        tracked_keypoints.extend(new_tracked_keypoints)
        return tracked_keypoints

    def _initialize_queries(self, keypoint):
        """
        Args:
            keypoint (np.ndarray): 初始关键点，形状为 (N, 2)。
        Returns:
            queries (torch.Tensor): 查询点张量，形状为 (1, N, 3)。
        """
        N = len(keypoint)
        frame_indices = torch.zeros((N, 1), dtype=torch.float32)  # 每个点对应的帧索引，全为 0
        keypoints_xy = torch.from_numpy(keypoint[:, ::-1].copy()).float()
        queries = torch.cat([frame_indices, keypoints_xy], dim=1).unsqueeze(0)  # (1, N, 3)
        return queries

    def _process_tracks(self, tracks, vis, keypoint, keypoint_info):
        """
        处理 CoTracker 的跟踪结果，返回最近 8 帧的关键点坐标，并存储对应时间帧索引。

        Args:
            tracks (torch.Tensor): (1, T, N, 2)，包含所有帧的跟踪坐标。
            vis (torch.Tensor): 可见性掩码，(1, T, N)。
            keypoint (np.ndarray or torch.Tensor): 初始关键点 (N, 3), 这里只用于取第三维作为Z来满足数据格式需要，后续需要修改
            keypoint_info (list): 存储关键点位置信息的列表。

        Returns:
            keypoints_numpy (np.ndarray): (8, N, 3)，包含最近 8 帧的 (x, y, z) 坐标。
        """
        if tracks is not None:
            T = tracks.shape[1]
            start_frame = max(0, T - 8)  # 计算起始帧索引，防止越界
            frame_indices = list(range(start_frame, T))  # T-8:T 的帧索引

            # 提取最近 8 帧的 (N, 2) 关键点坐标
            coords = tracks[:, -8:]  # (1, 8, N, 2)
            coords = coords.squeeze(0)  # (8, N, 2)

            keypoints_tensor = coords

            for t_idx, t in enumerate(frame_indices):  # 遍历实际时间帧索引
                for i in range(keypoints_tensor.shape[1]):  # 遍历 N 个关键点
                    keypoint_info.append({
                        "frame": t,  # 存储真实的帧索引 T-8:T
                        "object_id": i,  # 需要修改
                        "pixel_coords": (keypoints_tensor[t_idx, i, 0].item(), keypoints_tensor[t_idx, i, 1].item())
                    })

        else:
            # tracks为空则返回原始关键点
            if isinstance(keypoint, np.ndarray):
                keypoints_tensor = torch.from_numpy(keypoint).float().to(self.device)
            else:
                keypoints_tensor = keypoint.float().to(self.device)

            keypoints_tensor = keypoints_tensor[:, :2]
            for i in range(keypoints_tensor.shape[1]):  # 遍历 N 个关键点
                keypoint_info.append({
                    "frame": 8,
                    "object_id": i,
                    "pixel_coords": (keypoints_tensor[i, 0].item(), keypoints_tensor[i, 1].item())
                })

        keypoints_numpy = keypoints_tensor.cpu().numpy()  # (8, N, 3)
        return keypoints_numpy

    def _process_tracks_firstbatch(self, tracks, vis, keypoint, keypoint_info):
        if tracks is not None:
            frame_indices = list(range(0, 8))  # T-8:T 的帧索引

            # 提取最近 8 帧的 (N, 2) 关键点坐标
            coords = tracks[:, :8]  # (1, 8, N, 2)
            coords = coords.squeeze(0)  # (8, N, 2)

            keypoints_tensor = coords

            for t_idx, t in enumerate(frame_indices):  # 遍历实际时间帧索引
                for i in range(keypoints_tensor.shape[1]):  # 遍历 N 个关键点
                    keypoint_info.append({
                        "frame": t,  # 存储真实的帧索引 T-8:T
                        "object_id": i,  # 需要修改
                        "pixel_coords": (keypoints_tensor[t_idx, i, 0].item(), keypoints_tensor[t_idx, i, 1].item())
                    })
            # print(keypoint_info)
        keypoints_numpy = keypoints_tensor.cpu().numpy()  # (8, N, 3)
        return keypoints_numpy