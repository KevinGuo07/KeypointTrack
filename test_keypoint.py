import argparse
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from utils.keypoint_helper import KeypointHelper
import cv2
import pickle

# import pdb
# pdb.set_trace()

@dataclass
class Config:
    total_cat2idx_procthor_path: str = "./keguide/utils/total_cat2idx.json"
    total_cat2idx_alfred_path: str = "./keguide/utils/total_cat2idx_alfred.json"
    seg_model_path: str = "models/segmentation/maskrcnn_alfworld/mrcnn_alfred_objects_008_v3.pth"
    use_sem_seg: bool = True
    long_clip_path: str = "./checkpoints/longclip-B.pt"
    long_clip_gpu: int = 2  # 1
    sem_seg_gpu: int = 0
    cuda: bool = True

    # -------------vis_result-------------
    detic_mask: bool = True
    lisa_mask: bool = True
    base_save_path: str = "./keguide/examples/keypoint_test"
    detic_mask_base_save_path: str = "detic_mask"
    lisa_mask_base_save_path: str = "lisa_mask"
    affordance_map_base_save_path: str = "affordance_map"

    target_name = "block"


# 点云降采样
def downsample_point_cloud(point_cloud):
    # 计算需要的采样点数
    num_points = point_cloud.shape[0]

    # 随机选择点的索引
    sampled_indices = np.random.choice(num_points, 50000, replace=False)

    # 根据索引获取降采样后的点
    sampled_points = point_cloud[sampled_indices]

    return sampled_points


def main():
    args = Config
    keypoint_model = KeypointHelper(args)
    task_name = "dual_shoes_place" # block_handover, dual_shoes_place, dual_bottles_pick_easy
    
    # 假设这里就是你的 JSON 文件所在路径
    all_json_save_path = f"./keguide/block_pick_hard/all_guid_info.json"
    #all_json_save_path = "/data1/hydeng/Keguide_RoboTwin/eval_data/dp3/block_handover_L515_100/seed0/guid_info/guid_info_30164917048140/all_guid_info.json"

    # 读取 JSON
    if not os.path.exists(all_json_save_path):
        print(f"Error: JSON file not found: {all_json_save_path}")
        return
    
    with open(all_json_save_path, 'r') as f:
        data = json.load(f)  # data是一个 dict，例如 {"0": { ... }, "1": { ... }, ...}

    # 准备保存路径
    base_save_dir = args.base_save_path  # 保存路径示例: "./keguide/examples/keypoint_test"
    base_save_dir = os.path.join(base_save_dir, task_name)
    os.makedirs(base_save_dir, exist_ok=True)

    # 遍历每一个 step
    for step_idx_str in tqdm(sorted(data.keys(), key=lambda x: int(x))):
        guid_data = data[step_idx_str]

        # 从 guid_data 里取出关键数据
        # base_path = "/data1/hydeng/Keguide_RoboTwin/"
        base_path = "./keguide"
        rgb_npy_path = guid_data["rgb_image_npy"]
        rgb_npy_path = base_path + rgb_npy_path
        depth_npy_path = guid_data["depth_image_npy"]
        depth_npy_path = base_path + depth_npy_path
        masks_npy_path = guid_data["masks_npy"]
        masks_npy_path = base_path + masks_npy_path
        # pcd_npy_path = guid_data["point_cloud_npy"]
        # pcd_npy_path = base_path + pcd_npy_path
        info = guid_data["info"]  # 包含 fov, cameraHorizon, camera_world_xyz, rotation 等
        
        # 如果 segmentation 数据里包含多通道，需要你在生成时就区分好
        # 这里仅演示读取
        if not os.path.exists(rgb_npy_path) or not os.path.exists(depth_npy_path) or not os.path.exists(masks_npy_path):
            print(f"Some file not found for step {step_idx_str}, skip.")
            continue
        
        rgb_image = np.load(rgb_npy_path)       # shape: (H, W, 3), float 或 uint8
        depth_image = np.load(depth_npy_path)   # shape: (H, W), float depth
        masks = np.load(masks_npy_path)         # shape: (H, W)，如只含目标物体可直接使用
        # point_cloud = np.load(pcd_npy_path)     # shape: (N, 3)，float
        # print(point_cloud.shape)

        # 如果在生成数据时没有 label_info，就无法像原来那样通过 target_name 查找对应 mask
        # 这里假设你生成的 masks 已经是目标物体的单通道分割（即 1 表示目标物体，0 表示背景）
        # 如果需要更复杂的逻辑，需要在 JSON 或 npy 文件中保留物体 ID --> label 的映射
        # target_mask = (masks == 1).astype(np.uint8)  # or any ID you want

        # 构建 obs_dict
        obs_dict = {
            "rgb_image": rgb_image,
            "depth_image": depth_image,
            "info": info,
            "masks": masks,  # or target_mask 如果要针对特定目标物体
            "action": "Open", # 仅示例
            # "point_cloud": point_cloud
        }

        # 获取关键点
        keypoint_output_dict = keypoint_model.get_keypoint(obs_dict)
        keypoint_map = keypoint_output_dict["keypoint_map"]
        projected_image = keypoint_output_dict["projected_image"]

        # 对点云降采样
        keypoint_map = downsample_point_cloud(keypoint_map)

        # 组织保存路径
        step_save_dir = os.path.join(base_save_dir, f"step_{step_idx_str}")
        os.makedirs(step_save_dir, exist_ok=True)

        # 保存关键点
        keypoint_save_path = os.path.join(step_save_dir, "keypoint_map.npy")
        np.save(keypoint_save_path, keypoint_map)

        # 保存投影后的图像
        if projected_image is not None:
            # 如果 rgb_image 是 RGB，OpenCV写图需要 BGR，需转换一下
            if projected_image.shape[-1] == 3:
                projected_image = cv2.cvtColor(projected_image, cv2.COLOR_RGB2BGR)
            proj_img_save_path = os.path.join(step_save_dir, "keypoint_projected.png")
            cv2.imwrite(proj_img_save_path, projected_image)

        print(f"Step {step_idx_str}: keypoints saved to {keypoint_save_path}")

    print("All steps done.")

if __name__ == "__main__":
    main()
