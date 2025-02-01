import json
import math
import os
import re

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


def get_reference_features(keypoint_info, features_flat, H, W):
    reference_features_list = []
    # print(keypoint_info)
    for info in keypoint_info:
        if info["pixel_coords"] is None:
            ref_feat = torch.full_like(features_flat[0], float('nan'))
        else:
            col, row = info["pixel_coords"]
            idx = row * W + col
            ref_feat = features_flat[idx]  # shape: (D,)

        reference_features_list.append(ref_feat)

    reference_features = torch.stack(reference_features_list, dim=0)
    return reference_features


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

    def keypoint_track(self, dirs, base_path):
        image_files = sort_files_by_number(dirs["image"], prefix="rgb_")
        mask_files = sort_files_by_number(dirs["mask"], prefix="actor_seg_")
        obj_dicts_sorted = obj_dict_generate(dirs["info"], base_path)
        generate_point_clouds(obj_dicts_sorted, dirs["pointcloud"])
        point_files = sort_files_by_number(dirs["pointcloud"], prefix="point_")

        assert len(image_files) == len(point_files) == len(mask_files), "序列文件数量不一致！"

        tracked_keypoints = []
        keypoint_info = []
        window_frames = []

        for i, (img_file, point_file, mask_file) in enumerate(zip(image_files, point_files, mask_files)):
            image = load_image(os.path.join(dirs["image"], img_file))
            points = load_pointcloud(os.path.join(dirs["pointcloud"], point_file))
            mask = load_mask(os.path.join(dirs["mask"], mask_file))

            H, W, _ = image.shape

            if i == 0:
                keypoints, keypoint_info, candidate_pixels, features_flat = self.initialize_keypoint(image, points,
                                                                                                     mask, dirs["output"])
                reference_features = get_reference_features(keypoint_info, features_flat, H, W)
                queries = self._initialize_queries(keypoints)
                window_frames.append(image)
            else:
                if i % self.model.step == 0 and i != 0:
                    print(f"iteration time:",i)
                    print(keypoints.dtype)
                    keypoints, keypoint_info = self.cotrack_keypoints(image, keypoints, i, queries, window_frames, keypoint_info)
                    # 测试下面的更新是否有必要
                    queries = self._initialize_queries(keypoints)
                    print(f"keypoints of input {i} is tracked! image path is {img_file}")
                # keypoints, keypoint_info = self.initialize_keypoint(image, points, mask, output_dir)
                # start_time = time.time()
                # keypoints, keypoint_info, features_flat = self.track_keypoints(keypoints, reference_features, image, points, mask)

                # keypoints, keypoint_info, features_flat = self.track_keypoints(reference_features, image, points, mask)
                # reference_features = get_reference_features(keypoint_info, features_flat, H, W)

                # end_time = time.time()
                # print(f"Function run time: {end_time - start_time} seconds")
                else:
                    window_frames.append(image)

            tracked_keypoints.append((keypoints, keypoint_info))

        return tracked_keypoints

    def initialize_keypoint(self, image, points, mask, output_dir=None):
        keypoints, projected_img, candidate_pixels, features_flat = self.keypoint_proposer.get_keypoints(image, points,
                                                                                                         mask)
        real_keypoints, keypoint_info = self.keypoint_proposer.register_keypoints(keypoints, points, mask)

        if output_dir:
            save_image(projected_img, os.path.join(output_dir, "initial/projected_keypoints.jpg"))

        print("keypoint of first input image is initialized!")
        print("shape of image is:", image.shape, "shape of points is:", points.shape, "shape of mask is:", mask.shape)
        return real_keypoints, keypoint_info, candidate_pixels, features_flat

    # former version
    def track_keypoints(self,
                        reference_features,
                        image,
                        points,
                        mask,
                        cutoff_similarity=0.65,
                        top_k=50,
                        median_deviation=2,
                        topK_global=100):

        transformed_rgb, rgb, current_points, masks, shape_info = self.keypoint_proposer._preprocess(image, points,
                                                                                                     mask)
        features_flat = self.keypoint_proposer._get_features(transformed_rgb, shape_info)

        H = shape_info['img_h']
        W = shape_info['img_w']

        # occlusion judge
        # if
        features_flat = features_flat.to(self.device)
        features_flat_norm = F.normalize(features_flat, dim=1)
        reference_features = reference_features.to(self.device)
        reference_features_norm = F.normalize(reference_features, dim=1)

        similarities_matrix = reference_features_norm @ features_flat_norm.T  # (N, H*W)
        # print(similarities_matrix)
        updated_keypoints = []
        keypoint_info = []

        if isinstance(current_points, np.ndarray):
            current_points = torch.tensor(current_points, dtype=torch.float32)
        current_points = current_points.to(self.device)
        if isinstance(transformed_rgb, np.ndarray):
            transformed_rgb = torch.from_numpy(transformed_rgb).float()
        transformed_rgb = transformed_rgb.to(self.device)

        N = reference_features.shape[0]
        for i in range(N):

            sim_row = similarities_matrix[i]  # shape (H*W, )

            top_values, top_indices = sim_row.topk(topK_global)
            valid_mask = top_values > cutoff_similarity
            valid_top_values = top_values[valid_mask]
            valid_top_indices = top_indices[valid_mask]

            if valid_top_indices.numel() == 0:
                updated_keypoints.append(torch.tensor([float('nan')] * 3, device=self.device))
                keypoint_info.append({
                    "object_id": None,
                    "pixel_coords": None
                })
                print("non point is selected")
                continue

            final_top_k = min(top_k, valid_top_indices.shape[0])
            sorted_idx = torch.argsort(valid_top_values, descending=True)
            final_indices = valid_top_indices[sorted_idx[:final_top_k]]  # (final_top_k, )

            # test the coordinate
            matched_coords = torch.stack([
                final_indices % W,  # row
                final_indices // W  # col
            ], dim=1)

            median = torch.median(matched_coords, dim=0).values  # (2,)
            dev = torch.norm(matched_coords.float() - median.float(), dim=1)
            filtered_coords = matched_coords[dev < median_deviation]

            if filtered_coords.size(0) == 0:
                updated_keypoints.append(torch.tensor([float('nan')] * 3, device=self.device))
                keypoint_info.append({
                    "object_id": None,
                    "pixel_coords": None
                })
                print("all the points are filtered")
            else:
                avg_coords = torch.mean(filtered_coords.float(), dim=0).long()
                updated_3d = current_points[avg_coords[1], avg_coords[0]]
                updated_keypoints.append(updated_3d)
                object_id = mask[avg_coords[1]][avg_coords[0]]  # coord test

                keypoint_info.append({
                    "object_id": object_id,
                    "pixel_coords": (avg_coords[0].item(), avg_coords[1].item())
                })
                # print("keypoint id", object_id)

        updated_keypoints = torch.stack(updated_keypoints, dim=0)  # (N, 3)
        updated_keypoints_np = updated_keypoints.cpu().numpy()  # 转为 NumPy 数组

        smoothed_keypoints = uniform_filter1d(updated_keypoints_np, size=10, axis=0)
        smoothed_keypoints_tensor = torch.tensor(smoothed_keypoints, device=self.device)

        # updated_keypoints_cpu = updated_keypoints.cpu()

        return smoothed_keypoints_tensor, keypoint_info, features_flat

    # 此时可以拿到的输入有：上一帧的参考点坐标和对应的特征信息，当前帧的rgb、点云、mask
    # 考虑加入的输入：之前所有帧的track
    def cotrack_keypoints(self, image, keypoint, iteration, queries, window_frames, keypoint_info):
        """
        Args:
            image (np.ndarray): 当前帧图像，形状为 (H, W, 3)。
            keypoint (np.ndarray): 初始关键点，形状为 (N, 3)，包含 x, y, z。
            iteration (int): 当前迭代次数，1 表示初始化帧。
        Returns:
            keypoints_tensor (torch.Tensor): 当前帧的关键点坐标，形状为 (N, 3)。
            keypoint_info (list): 包含关键点的位置信息的字典列表。
        """


        # 调整输入图片格式，对应video_chunk
        queries = queries.to(self.device)
        if iteration == self.model.step:
            # debug: queries格式没问题

            # print(f"query {iteration}: type={type(queries)}, shape= {queries.shape}, {queries}, dtype={queries.dtype}")

            # debug: tracks格式没问题
            tracks, vis = self._process_step(
                window_frames,
                is_first_step=True,
                query=queries,
                grid_query_frame=0,
            )
            window_frames.append(image)
            keypoints_tensor = self._process_tracks(tracks, vis, keypoint, keypoint_info)
            print(f"Frame {iteration}: type={type(tracks)}")  # ,shape={tracks.shape}, dtype={tracks.dtype}
        else:
            tracks, vis = self._process_step(
                window_frames,
                is_first_step=False,
                query=queries,
                grid_query_frame=0,  # 检查是否可以不要
            )
            window_frames.append(image)
            keypoints_tensor = self._process_tracks(tracks, vis, keypoint, keypoint_info)
            print(f"Frame {iteration}: type={type(tracks)}, shape={tracks.shape}, dtype={tracks.dtype}")
        # to be solved: z坐标值的更新
        # keypoints_tensor = self._process_tracks(tracks, vis, keypoint, keypoint_info)

        return keypoints_tensor, keypoint_info

    def _initialize_queries(self, keypoint):
        """
        Args:
            keypoint (np.ndarray): 初始关键点，形状为 (N, 3)。
        Returns:
            queries (torch.Tensor): 查询点张量，形状为 (1, N, 3)。
        """
        keypoints_2d = keypoint[:, :2]  # 提取 x, y 坐标
        N = len(keypoints_2d)
        frame_indices = torch.zeros((N, 1), dtype=torch.float32)  # 每个点对应的帧索引，全为 0
        print(keypoints_2d.dtype)
        keypoints_xy = torch.from_numpy(keypoints_2d).float()  # 转为张量
        queries = torch.cat([frame_indices, keypoints_xy], dim=1).unsqueeze(0)  # 拼接为 (1, N, 3)
        return queries

    def _process_tracks(self, tracks, vis, keypoint, keypoint_info):
        """
        Args:
            tracks (torch.Tensor or None): 跟踪结果，形状为 (1, T, N, 2)。
            vis (torch.Tensor): 可见性掩码，形状为 (1, T, N)。
            keypoint (np.ndarray): 初始关键点，形状为 (N, 3)。
            keypoint_info (list): 存储关键点位置信息的列表。
        Returns:
            keypoints_tensor (torch.Tensor): 当前帧的关键点坐标，形状为 (N, 3)。
        """
        if tracks is not None:
            coords = tracks[0, 0]  # 提取当前帧的关键点坐标 (N, 2)
            visible_mask = vis[0, 0].cpu().numpy()  # 提取可见性掩码 (N,)
            keypoint_z = keypoint[:, 2:3]  # 提取 z 坐标，形状为 (N, 1)

            # 确保 keypoint_z 是 Tensor 且与 coords 在同一设备上
            keypoint_z_tensor = torch.from_numpy(keypoint_z).to(coords.device)

            keypoints_tensor = torch.cat([coords, keypoint_z_tensor], dim=1)  # (N, 3)

            for i, coord in enumerate(coords):
                keypoint_info.append({
                    "object_id": i,
                    "pixel_coords": (coord[0].item(), coord[1].item())
                })
            keypoints_numpy = keypoints_tensor.cpu().numpy().astype(np.float16)
        else:
            # 如果 tracks 为 None，使用初始关键点作为 fallback
            keypoints_tensor = torch.from_numpy(keypoint).float().to(self.device)
            keypoints_tensor = keypoints_tensor[:, :3]  # 只保留 (x, y, z)

            for i, coord in enumerate(keypoints_tensor):
                keypoint_info.append({
                    "object_id": i,
                    "pixel_coords": (coord[0].item(), coord[1].item())
                })
            keypoints_numpy = keypoints_tensor.cpu().numpy().astype(np.float16)

        return keypoints_numpy
