import numpy as np
# from models.Detic.segmentation_helper_procthor_detic import SemgnetationHelperProcThorDetic
import random
from typing import Tuple
from collections import deque
from typing import Optional, Sequence, cast

from tqdm import tqdm
import open3d as o3d

import torch
import json
import math
# from PIL import Image
# from utils.keypoint_utils import KeypointProposer
from utils.keypoint_utils import KeypointProposer

# from kmeans_pytorch import kmeans
from sklearn.cluster import MeanShift
import open3d as o3d

# 配置环境参数
AGENT_STEP_SIZE = 0.25
RECORD_SMOOTHING_FACTOR = 1
CAMERA_HEIGHT_OFFSET = 0.75
VISIBILITY_DISTANCE = 25


class KeypointHelper(object):
    def __init__(self, args):
        self.args = args
        # load Detic
        # self.seg = SemgnetationHelperProcThorDetic(self)

        # object_dict
        self.total_cat2idx = json.load(open(args.total_cat2idx_procthor_path))
        self.total_idx2cat = {}
        for cat, index in self.total_cat2idx.items():
            self.total_idx2cat[index] = cat

        # load keypoint
        config = {
            'device': "cuda:0",
            'bounds_min': [-1.2, -1.2, -1.2], # [0.0, 0.0, 0.0]
            'bounds_max': [1.95, 1.95, 1.95], # [30.0, 30.0, 30.0]
            'min_dist_bt_keypoints': 0.05,
            'max_mask_ratio': 0.5,
            'num_candidates_per_mask': 5,
            'seed': 42
        }

        self.config = config
        self.keypoint_proposer = KeypointProposer(config) # 从RGB和mask中提取关键点


    def get_keypoint(self, obs_dict):
        rgb = obs_dict["rgb_image"]
        masks = obs_dict["masks"]
        depth_image = obs_dict["depth_image"]
        # info = obs_dict["info"]

        points = self.get_point_cloud(obs_dict)
        # points = obs_dict["point_cloud"]
        points = points.reshape([rgb.shape[0], rgb.shape[1], -1])
        candidate_keypoints, projected_image, candidate_pixels = self.keypoint_proposer.get_keypoints(
            rgb,
            points,
            masks
        )

        if candidate_keypoints.size == 0:
            print("没有找到任何候选关键点！")
            return {"keypoint_map": np.array([]), "projected_image": None}
    
        # 生成关键点的mask
        keypoint_mask = np.zeros_like(masks)
        for keypoint_count, pixel in enumerate(candidate_pixels):
            # draw a box
            box_width = 30
            box_height = 30
            bbox_min = [pixel[1] - box_width // 2, pixel[0] - box_height // 2]
            bbox_max = [pixel[1] + box_width // 2, pixel[0] + box_height // 2]
            keypoint_mask[bbox_min[1]:bbox_max[1], bbox_min[0]:bbox_max[0]] = 1

        # Lift to 3D
        affordance_score = keypoint_mask.reshape([-1, 1])
        points = points.reshape([-1, 3])
        print(affordance_score.shape)
        print(points.shape)

        # (x, y, z, affordance_score)
        world_space_point_cloud = np.concatenate((points, affordance_score), axis=1)

        mask = np.ones(depth_image.shape, dtype=bool)
        # rgb_color = rgb_image[:, :, ::-1]
        rgb_image = rgb[:, :, ::-1] # BGR to RGB
        r_color = rgb_image[:, :, 0][mask][None, :]
        g_color = rgb_image[:, :, 1][mask][None, :]
        b_color = rgb_image[:, :, 2][mask][None, :] 
        rgb_color = np.concatenate((r_color, g_color, b_color), axis=0).T

        world_space_point_cloud = np.concatenate((world_space_point_cloud, rgb_color), axis=1)
        keypoint_output_dict = {"keypoint_map": world_space_point_cloud,
                                  "projected_image": projected_image}

        return keypoint_output_dict


    # def get_point_cloud(self, depth_one, affordance_score, info_dict_one, point_only=False):
    #     fov = info_dict_one["fov"]
    #     cameraHorizon = info_dict_one["cameraHorizon"]
    #     camera_world_xyz = info_dict_one["camera_world_xyz"]
    #     if not isinstance(camera_world_xyz, np.ndarray):
    #         camera_world_xyz = np.asarray(camera_world_xyz)
    #     rotation = info_dict_one["rotation"]

    #     # 映射点云
    #     camera_space_point_cloud = self.cpu_only_depth_frame_to_camera_space_xyz(depth_one, mask=None, fov=fov)
    #     partial_point_cloud = self.cpu_only_camera_space_xyz_to_world_xyz(camera_space_point_cloud,
    #         camera_world_xyz, rotation, cameraHorizon)

    #     if not point_only:
    #         select_mask = np.ones(depth_one.shape, dtype=bool)
    #         affordance_score = affordance_score[select_mask][None, :]
    #         world_space_point_cloud = np.concatenate((partial_point_cloud, affordance_score), axis=0)
    #         world_space_point_cloud = world_space_point_cloud.T
    #         world_space_point_cloud[:, [0, 1, 2, 3]] = world_space_point_cloud[:, [0, 2, 1, 3]]
    #         world_space_point_cloud = world_space_point_cloud.astype(np.float16)
    #     else:
    #         world_space_point_cloud = partial_point_cloud.T
    #         world_space_point_cloud[:, [0, 1, 2]] = world_space_point_cloud[:, [0, 2, 1]]
    #         world_space_point_cloud = world_space_point_cloud.astype(np.float16)

    #     return world_space_point_cloud


    def get_point_cloud(self, obs_dict):
        """
        """
        # 读出 RGB 与 Position（[H, W, 4]）
        rgb_image = obs_dict["rgb_image"]
        depth_image = obs_dict["depth_image"]
        info = obs_dict["info"]

        model_matrix = info.get("model_matrix", None)
        if model_matrix is None:
            raise ValueError("info_dict 中缺少 'model_matrix' 信息。请确保在构建 obs_dict 时包含 'model_matrix'。")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_matrix_t = torch.tensor(model_matrix, dtype=torch.float32).to(device)

        H,W = depth_image.shape

        y, x = np.indices((H, W))
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)
        z = torch.from_numpy(depth_image).float().to(device) / 1000.0  # 转换为米，(H, W)


        fov = info["fov"]
        fov_rad = fov / 180.0 * math.pi
        focal_length = W / (2.0 * math.tan(fov_rad / 2.0))

        x_cam = (x - W / 2.0) * z / focal_length
        y_cam = (y - H / 2.0) * z / focal_length
        z_cam = z
        points_cam = torch.stack((x_cam, y_cam, z_cam), dim=2)  # (H, W, 3)
        points_cam_flat = points_cam.view(-1, 3).unsqueeze(0)  # (1, H*W, 3)

        rotation = model_matrix_t[:3, :3].transpose(0, 1).unsqueeze(0)  # (1, 3, 3)
        translation = model_matrix_t[:3, 3].unsqueeze(0)                # (1, 3)
        points_world = torch.bmm(points_cam_flat, rotation) + translation  # (1, H*W, 3)
        points_world_np = points_world.squeeze(0).cpu().numpy()
        
        
        # # 创建点云对象
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_world_np)

        # # 创建参考坐标系
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # # 创建点云可视化窗口并显示点云和坐标系
        # o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="Point Cloud with Reference Frame", width=800, height=600, left=50, top=50)


        return points_world_np.astype(np.float16)

