import argparse
import os
import sys

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
    total_cat2idx_procthor_path: str = "./utils/total_cat2idx.json"
    total_cat2idx_alfred_path: str = "./utils/total_cat2idx_alfred.json"
    seg_model_path: str = "models/segmentation/maskrcnn_alfworld/mrcnn_alfred_objects_008_v3.pth"
    use_sem_seg: bool = True
    long_clip_path: str = "./checkpoints/longclip-B.pt"
    long_clip_gpu: int = 2  # 1
    sem_seg_gpu: int = 0
    cuda: bool = True

    # -------------vis_result-------------
    detic_mask: bool = True
    lisa_mask: bool = True
    base_save_path: str = "./examples/keypoint_test"
    detic_mask_base_save_path: str = "detic_mask"
    lisa_mask_base_save_path: str = "lisa_mask"
    affordance_map_base_save_path: str = "affordance_map"

    target_name = "Sofa"


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
    # 初始化keypoint
    keypoint_model = KeypointHelper(args)

    # 配置文件路径
    image_base_path = "/data1/wzy/vision_dataset/nav_inter_dataset_test/rgb"
    depth_base_path = "/data1/wzy/vision_dataset/nav_inter_dataset_test/depth"
    info_base_path = "/data1/wzy/vision_dataset/nav_inter_dataset_test/info"
    mask_base_path = "/data1/wzy/vision_dataset/nav_inter_dataset_test/mask"
    ori_image_file_list = os.listdir(image_base_path) # 获取RGB图像列表

    # 获取house_id
    house_id_dict = {}
    for image_one in ori_image_file_list:
        image_one_list = image_one.split("_")
        image_house_id = image_one_list[0] + "_" + image_one_list[1]
        if image_house_id in house_id_dict:
            house_id_dict[image_house_id].append(image_one)
        else:
            house_id_dict[image_house_id] = []
            house_id_dict[image_house_id].append(image_one)

    # 选择需要处理的house
    select_house_list = ["house_65"]

    for house_id, image_file_list in house_id_dict.items():
        if house_id not in select_house_list:
            continue

        # 设定保存路径
        house_save_path = os.path.join(args.base_save_path, house_id + "_" + args.target_name)
        affordance_map_base_save_path = os.path.join(house_save_path, args.affordance_map_base_save_path)
        detic_mask_base_save_path = os.path.join(house_save_path, args.detic_mask_base_save_path)
        lisa_mask_base_save_path = os.path.join(house_save_path, args.lisa_mask_base_save_path)

        if not os.path.exists(affordance_map_base_save_path):
            os.makedirs(affordance_map_base_save_path)
        if not os.path.exists(detic_mask_base_save_path):
            os.makedirs(detic_mask_base_save_path)
        if not os.path.exists(lisa_mask_base_save_path):
            os.makedirs(lisa_mask_base_save_path)

        for image_name_one in tqdm(image_file_list):
            image_path = os.path.join(image_base_path, image_name_one)
            depth_path = os.path.join(depth_base_path, image_name_one.replace(".png", ".npz"))
            info_path = os.path.join(info_base_path, image_name_one.replace(".png", ".pkl"))
            mask_path = os.path.join(mask_base_path, image_name_one.replace(".png", ".npz"))

            rgb_image = cv2.imread(image_path)
            depth = np.load(depth_path)["depth_image"]
            masks = np.load(mask_path)["mask"]
            with open(info_path, 'rb') as file:
                info = pickle.load(file)

            # 从info中获取目标mask
            target_mask = None
            label_info = info["label_info"]
            print(label_info)
            for mask_id, mask in enumerate(masks):
                class_name = label_info[mask_id]["class_name"]
                if class_name == args.target_name:
                    target_mask = mask
                    break
            if target_mask is None:
                # 没有找到目标mask
                continue

            obs_dict = {"rgb_image": rgb_image,
                        "depth_image": depth,
                        "info": info,
                        "masks": target_mask,
                        "action": "Open"}

            # 开始调用keypoint模型获取关键点地图与投影图像
            keypoint_output_dict = keypoint_model.get_keypoint(obs_dict)
            keypoint_map = keypoint_output_dict["keypoint_map"]
            projected_image = keypoint_output_dict["projected_image"]

            # 进行降采样保存
            keypoint_map = downsample_point_cloud(keypoint_map)

            if not os.path.exists(affordance_map_base_save_path):
                os.makedirs(affordance_map_base_save_path)
            save_path = os.path.join(affordance_map_base_save_path, image_name_one.replace(".png", ".npy"))
            np.save(save_path, keypoint_map)

            # 保存不同的mask
            # detic_mask_base_save_path = args.detic_mask_base_save_path
            if projected_image is not None:
                if not os.path.exists(detic_mask_base_save_path):
                    os.makedirs(detic_mask_base_save_path)
                detic_mask_save_path = os.path.join(detic_mask_base_save_path, image_name_one)
                cv2.imwrite(detic_mask_save_path, projected_image)

if __name__ == "__main__":
    main()
