import argparse
import os
import re
import numpy as np
import torch

from cotracker.predictor import CoTrackerOnlinePredictor
from utils.cotrack_utils import KeypointTracker, obj_dict_generate, generate_point_clouds
from utils.preprocess import load_image, load_pointcloud, load_mask, save_image

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def sort_files_by_number(directory, prefix):
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]

    def extract_number(filename):
        match = re.search(fr'{prefix}(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    return sorted(files, key=extract_number)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="C:/Users/11760/Desktop/dissertation/KeypointTrack/checkpoints/scaled_online.pth",
        # path to your checkpoint path
        help="CoTracker model parameters",
    )

    parser.add_argument(
        "--base_dir",
        default="C:/Users/11760/Desktop/dissertation/KeypointTrack/dual_shoes_place",  # path to your file path
        help="Base directory for input and output data",
    )

    args = parser.parse_args()

    dirs = {
        "image": os.path.join(args.base_dir, "image"),
        "pointcloud": os.path.join(args.base_dir, "pointcloud_npy"),
        "mask": os.path.join(args.base_dir, "seg_npy"),
        "output": os.path.join(args.base_dir, "output"),
        "info": os.path.join(args.base_dir, "all_guid_info.json")
    }

    os.makedirs(dirs["output"], exist_ok=True)

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    tracker = KeypointTracker(model=model)

    project_path = os.path.dirname(args.base_dir)
    task_path = args.base_dir
    image_files = sort_files_by_number(dirs["image"], prefix="rgb_")
    mask_files = sort_files_by_number(dirs["mask"], prefix="actor_seg_")
    obj_dicts_sorted = obj_dict_generate(dirs["info"], project_path)
    generate_point_clouds(obj_dicts_sorted, dirs["pointcloud"])
    point_files = sort_files_by_number(dirs["pointcloud"], prefix="point_")

    assert len(image_files) == len(point_files) == len(mask_files), "序列文件数量不一致！"

    tracked_keypoints = []
    window_frames = []
    keypoint_info = []
    is_first_step = True
    for i, (img_file, point_file, mask_file) in enumerate(zip(image_files, point_files, mask_files)):
        image = load_image(os.path.join(dirs["image"], img_file))
        points = load_pointcloud(os.path.join(dirs["pointcloud"], point_file))
        mask = load_mask(os.path.join(dirs["mask"], mask_file))

        H, W, _ = image.shape

        if i == 0:
            keypoints, keypoints_points, features_flat = tracker.initialize_keypoint(image, points,
                                                                                  mask,
                                                                                  task_path)
            queries = tracker._initialize_queries(keypoints)
            queries = queries.to(tracker.device)
        else:
            if i == 8:
                pred_tracks, pred_visibility = tracker._process_step(
                    window_frames,
                    is_first_step=is_first_step,
                    query=queries,
                    grid_query_frame=0,  # 检查是否可以不要
                )
                is_first_step = False
            if i > 16:
                pred_tracks, pred_visibility = tracker._process_step(
                    window_frames,
                    is_first_step=is_first_step,
                    query=queries,
                    grid_query_frame=0,  # 检查是否可以不要
                )
                # print(pred_tracks.shape)
                print(pred_tracks[0, pred_tracks.shape[1]-1, 0:1, :])
        window_frames.append(image)

    print(pred_tracks.shape)





if __name__ == "__main__":
    main()

