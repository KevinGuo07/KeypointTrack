import torch
import os
import re
import argparse
from utils.preprocess import load_image
from utils.video_utils import create_video_from_images
from cotracker.predictor import CoTrackerOnlinePredictor
from utils.cotrack_utils import KeypointTracker, save_and_show_keypoints

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
        default=None,
        help="CoTracker model parameters",
    )

    parser.add_argument(
        "--base_dir",
        default="C:/Users/11760/Desktop/dissertation/keguide/block_pick_hard",
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
    tracked_keypoints = tracker.keypoint_track(dirs["image"], dirs["mask"], dirs["info"], dirs["pointcloud"], dirs["output"])
    image_files = sort_files_by_number(dirs["image"], "rgb_")

    for i, (keypoints, keypoint_info) in enumerate(tracked_keypoints):
        image_path = os.path.join(dirs["image"], image_files[i])
        image = load_image(image_path)
        print(f"Processing image file: {os.path.basename(image_path)}")
        save_and_show_keypoints(image, keypoint_info, image_path, dirs["output"], i)

    output_video_path = os.path.join(args.base_dir, "keypoint_tracking.mp4")
    save_dir = sort_files_by_number(os.path.join(args.base_dir, "output"), "keypoints_")
    save_dir = [os.path.join(os.path.join(args.base_dir, "output"), filename) for filename in save_dir]
    create_video_from_images(save_dir, output_video_path)


if __name__ == "__main__":
    main()

