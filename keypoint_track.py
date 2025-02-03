import json
import math
import numpy as np
import torch
import yaml
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.keypoint_utils import KeypointProposer
from utils.preprocess import load_image, load_pointcloud, load_mask, save_image
from utils.video_utils import create_video_from_images
import os
import re
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d


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


class KeypointTracker:
    def __init__(self, device=None):
        kp_config = get_config(config_path="kp_config.yaml")
        self.keypoint_proposer = KeypointProposer(kp_config['keypoint_proposer'])
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def keypoint_track(self, image_seq_dir, depth_dir, mask_seq_dir, output_dir):

        image_files = sort_files_by_number(image_seq_dir, prefix="rgb_")
        mask_files = sort_files_by_number(mask_seq_dir, prefix="actor_seg_")

        obj_dicts_sorted = self.obj_dict_generate("C:/Users/11760/Desktop/dissertation/KeypointTrack/dual_shoes_place"
                                                  "/all_guid_info.json")
        # print(obj_dicts_sorted)
        # depth_image = obj_dicts_sorted["depth_image"]
        point_seq_dir = "C:/Users/11760/Desktop/dissertation/KeypointTrack/dual_shoes_place/pointcloud_npy"

        self.generate_point_clouds(obj_dicts_sorted, point_seq_dir)
        point_files = sort_files_by_number(point_seq_dir, prefix="point_")

        assert len(image_files) == len(point_files) == len(mask_files), "序列文件数量不一致！"

        tracked_keypoints = []
        for i, (img_file, point_file, mask_file) in enumerate(zip(image_files, point_files, mask_files)):
            image = load_image(os.path.join(image_seq_dir, img_file))
            points = load_pointcloud(os.path.join(point_seq_dir, point_file))
            mask = load_mask(os.path.join(mask_seq_dir, mask_file))

            H, W, _ = image.shape

            if i == 0:
                keypoints, keypoint_info, candidate_pixels, features_flat = self.initialize_keypoint(image, points, mask, output_dir)
                reference_features = self.get_reference_features(keypoint_info, features_flat, H, W)

            else:
                # keypoints, keypoint_info = self.initialize_keypoint(image, points, mask, output_dir)
                # start_time = time.time()
                keypoints, keypoint_info,features_flat = self.track_keypoints(keypoints, reference_features, image, points, mask)
                # print("跟踪后的keypoint为", keypoints)
                # print("跟踪后的keypoint info为", keypoint_info)
                reference_features = self.get_reference_features(keypoint_info, features_flat, H, W)
                # end_time = time.time()
                # print(f"Function run time: {end_time - start_time} seconds")
                print(f"keypoints of input {i} is tracked! image path is {img_file}")

            tracked_keypoints.append((keypoints, keypoint_info))

        return tracked_keypoints

    def initialize_keypoint(self, image, points, mask, output_dir=None):
        keypoints, projected_img, candidate_pixels, features_flat = self.keypoint_proposer.get_keypoints(image, points, mask)
        real_keypoints, keypoint_info = self.keypoint_proposer.register_keypoints(keypoints, points, mask)

        if output_dir:
            save_image(projected_img, os.path.join(output_dir, "initial/projected_keypoints.jpg"))
        print(candidate_pixels)
        print("keypoint of first input image is initialized!")
        return real_keypoints, keypoint_info, candidate_pixels, features_flat

    def get_reference_features(self, keypoint_info, features_flat, H, W):
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

    # former version
    def track_keypoints(self,
                        keypoints_pixel_coords,
                        reference_features,
                        image,
                        points,
                        mask,
                        cutoff_similarity=0.65,
                        top_k=50,
                        median_deviation=2,
                        topK_global=100):

        transformed_rgb, rgb, current_points, masks, shape_info = self.keypoint_proposer._preprocess(image, points, mask)
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
                object_id = mask[avg_coords[1]][avg_coords[0]]  #coord test

                keypoint_info.append({
                    "object_id": object_id,
                    "pixel_coords": (avg_coords[0].item(), avg_coords[1].item())
                })
                # print("keypoint id", object_id)

        updated_keypoints = torch.stack(updated_keypoints, dim=0)  # (N, 3)
        updated_keypoints_np = updated_keypoints.cpu().numpy()

        smoothed_keypoints = uniform_filter1d(updated_keypoints_np, size=10, axis=0)
        smoothed_keypoints_tensor = torch.tensor(smoothed_keypoints, device=self.device)
        print("跟踪后的keypoint为", smoothed_keypoints_tensor)
        # updated_keypoints_cpu = updated_keypoints.cpu()

        return smoothed_keypoints_tensor, keypoint_info, features_flat

    def obj_dict_generate(self, all_json_save_path):
        """生成包含所有帧信息的字典"""
        if not os.path.exists(all_json_save_path):
            raise FileNotFoundError(f"JSON file not found: {all_json_save_path}")

        with open(all_json_save_path, 'r') as f:
            data = json.load(f)

        obs_dicts = {}
        for step_idx_str in tqdm(sorted(data.keys(), key=lambda x: int(x))):
            guid_data = data[step_idx_str]
            base_path = "C:/Users/11760/Desktop/dissertation/KeypointTrack/"

            paths = {
                "rgb": os.path.join(base_path, guid_data["rgb_image_npy"]),
                "depth": os.path.join(base_path, guid_data["depth_image_npy"]),
                "masks": os.path.join(base_path, guid_data["masks_npy"]),
            }

            if not all(os.path.exists(p) for p in paths.values()):
                print(f"Files missing for step {step_idx_str}, skipping.")

                for key, path in paths.items():
                    if not os.path.exists(path):
                        print(f"  Missing {key}: {path}")
                continue

            obs_dicts[int(step_idx_str)] = {
                "rgb_image": np.load(paths["rgb"]),
                "depth_image": np.load(paths["depth"]),
                "masks": np.load(paths["masks"]),
                "info": guid_data["info"],
                "frame_id": step_idx_str
            }

        return obs_dicts

    def generate_point_clouds(self, obj_dicts, point_seq_dir):
        """为所有帧生成点云"""
        for frame_id, obs_dict in obj_dicts.items():
            try:
                self.get_point_cloud(obs_dict, point_seq_dir)
            except Exception as e:
                print(f"Error processing frame {frame_id}: {e}")

    def get_point_cloud(self, obs_dict, point_seq_dir):

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


# 主函数
def main():
    base_dir = "C:/Users/11760/Desktop/dissertation/KeypointTrack/dual_shoes_place"
    dirs = {
        "image": os.path.join(base_dir, "image"),
        "pointcloud": os.path.join(base_dir, "pointcloud_npy"),
        "mask": os.path.join(base_dir, "seg_npy"),
        "output": os.path.join(base_dir, "output"),
    }
    # only used when its raw data
    # sort_and_reorder_files(dirs["image"], dirs["pointcloud"], dirs["mask"])
    os.makedirs(dirs["output"], exist_ok=True)

    tracker = KeypointTracker()
    tracked_keypoints = tracker.keypoint_track(dirs["image"], dirs["pointcloud"], dirs["mask"], dirs["output"])
    image_files = sort_files_by_number(dirs["image"], "rgb_")

    for i, (keypoints, keypoint_info) in enumerate(tracked_keypoints):

        image_path = os.path.join(dirs["image"], image_files[i])
        image = load_image(image_path)
        print(f"Processing image file: {os.path.basename(image_path)}")

        '''
        image draw with keypoint and save to certain path
        '''
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

            if pixel_coords is not None:
                # 根据 object_id 从 color_dict 中获取对应颜色
                color = color_dict[obj_id]
                plt.scatter(pixel_coords[0], pixel_coords[1], s=50, edgecolors='white', color=color,
                            label=f"bottle_id: {obj_id}")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=os.path.basename(image_path))
        plt.axis('off')

        plt.savefig(os.path.join(dirs["output"], f"keypoints_{i}.jpg"), bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
    base_dir = "C:/Users/11760/Desktop/dissertation/KeypointTrack/dual_shoes_place/"
    output_video_path = os.path.join(base_dir, "keypoint_track.mp4")

    save_dir = sort_files_by_number(os.path.join(base_dir, "output"), "keypoints_")
    save_dir = [os.path.join(os.path.join(base_dir, "output"), filename) for filename in save_dir]
    create_video_from_images(save_dir, output_video_path)
