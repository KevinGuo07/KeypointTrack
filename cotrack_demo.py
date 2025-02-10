import argparse
import os
import torch
import time

from cotracker.predictor import CoTrackerOnlinePredictor
from utils.video_utils import create_video_from_images
from utils.cotrack_utils import *
from utils.preprocess import *

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="C:/Users/11760/Desktop/dissertation/KeypointTrack/checkpoints/scaled_online.pth",   # checkpoint path
        help="CoTracker model parameters",
    )

    parser.add_argument(
        "--base_dir",
        default="C:/Users/11760/Desktop/dissertation/KeypointTrack/block_pick_hard",  # path to your file path
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
    buffer_image=[]
    keypoint_info = []
    is_first_step = True
    start_time=time.time()
    for i, (img_file, point_file, mask_file) in enumerate(zip(image_files, point_files, mask_files)):
        image = load_image(os.path.join(dirs["image"], img_file))
        points = load_pointcloud(os.path.join(dirs["pointcloud"], point_file))
        mask = load_mask(os.path.join(dirs["mask"], mask_file))

        H, W, _ = image.shape

        if i == 0:
            start_time1 = time.time()
            keypoints, keypoints_points, features_flat = tracker.initialize_keypoint(image, points,
                                                                                     mask,
                                                                                     task_path)
            end_time1 = time.time()
            print(f"initialize time is {end_time1 - start_time1} seconds")
            keypoints_flipped = keypoints[:, [1, 0]]
            draw_points(image, keypoints_flipped, dirs["output"], i)
            queries = (tracker._initialize_queries(keypoints)).to(tracker.device)
            # queries = queries.to(tracker.device)
            # 第0帧重复8遍，以前置跟踪窗口
            for i in range(model.step):
                window_frames.append(image)
            pred_tracks, pred_visibility = tracker._process_step(
                window_frames,
                is_first_step=is_first_step,
                query=queries,
                grid_query_frame=0,
            )
            is_first_step = False
            window_frames.append(image)
        else:
            window_frames.append(image)
            if i in range(1, model.step):
                buffer_image.append(image)
            real_index = model.step - 1
            if i == real_index:
                pred_tracks, pred_visibility = tracker._process_step(
                    window_frames,
                    is_first_step=is_first_step,
                    query=queries,
                    grid_query_frame=0,
                )
                for j in range(1, model.step):
                    selected_point = pred_tracks[:, j, :, :].squeeze(0).cpu().numpy()
                    selected_vis = pred_visibility[:, j, :].squeeze(0).cpu().numpy()

                    draw_points(buffer_image[j-1], selected_point, dirs["output"], j)

            elif i > real_index:
                pred_tracks, pred_visibility = tracker._process_step(
                    window_frames,
                    is_first_step=is_first_step,
                    query=queries,
                    grid_query_frame=0,
                )
                latest_point=pred_tracks[:, -1, :, :].squeeze(0).cpu().numpy()
                draw_points(image, latest_point, dirs["output"], i)
    end_time=time.time()
    print(f"iteration time is {end_time-start_time} seconds")
    output_video_path = os.path.join(args.base_dir, "keypoint_cotrack.mp4")
    save_dir = sort_files_by_number(os.path.join(args.base_dir, "output"), "keypoints_")
    save_dir = [os.path.join(os.path.join(args.base_dir, "output"), filename) for filename in save_dir]
    create_video_from_images(save_dir, output_video_path)


if __name__ == "__main__":
    main()
