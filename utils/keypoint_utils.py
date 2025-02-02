import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from sklearn.cluster import MeanShift


def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """
    Filter points by taking only points within workspace bounds.
    """

    assert points.shape[1] == 3, "points must be (N, 3)"
    bounds_min = bounds_min.copy()
    bounds_max = bounds_max.copy()
    if not strict:
        bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
    within_bounds_mask = (
            (points[:, 0] >= bounds_min[0])
            & (points[:, 0] <= bounds_max[0])
            & (points[:, 1] >= bounds_min[1])
            & (points[:, 1] <= bounds_max[1])
            & (points[:, 2] >= bounds_min[2])
            & (points[:, 2] <= bounds_max[2])
    )
    return within_bounds_mask


class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self.device)
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.mean_shift = MeanShift(bandwidth=self.config['min_dist_bt_keypoints'], bin_seeding=True, n_jobs=32)
        self.patch_size = 14  # dinov2
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

    def get_keypoints(self, rgb, points, masks):
        # preprocessing
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)
        num_arrays = len(masks)

        # print(f"Number of arrays in masks after preprocess: {num_arrays}")

        # get features
        features_flat = self._get_features(transformed_rgb, shape_info)
        # for each mask, cluster in feature space to get meaningful regions, and uske their centers as keypoint candidates
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(points, features_flat,
                                                                                                  masks)

        # print(f"candidate_pixels shape before exclude: {candidate_pixels.shape}")
        # print(f"candidate_pixels: {candidate_pixels}")
        # exclude keypoints that are outside of the workspace

        within_space = filter_points_by_bounds(candidate_keypoints, self.bounds_min, self.bounds_max, strict=True)
        candidate_keypoints = candidate_keypoints[within_space]
        candidate_pixels = candidate_pixels[within_space]
        candidate_rigid_group_ids = candidate_rigid_group_ids[within_space]

        # merge close points by clustering in cartesian space
        # merged_indices = self._merge_clusters(candidate_keypoints)
        # candidate_keypoints = candidate_keypoints[merged_indices]
        # candidate_pixels = candidate_pixels[merged_indices]
        # candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]

        print(f"candidate_pixels shape: {candidate_pixels.shape}")
        # print(f"candidate_pixels: {candidate_pixels}")
        # sort candidates by locations
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]

        # project keypoints to image space
        projected = self._project_keypoints_to_img(rgb, candidate_pixels, candidate_rigid_group_ids, masks,
                                                   features_flat)

        return candidate_keypoints, projected, candidate_pixels, features_flat

    def _preprocess(self, rgb, points, masks):
        # convert masks to binary masks

        unique_ids = np.unique(masks)  # 获取所有独特的mask值
        masks = [masks == uid for uid in unique_ids if uid >= 66]  # 为每个大于等于66的独特值创建一个二值掩码
        # masks = [masks == uid for uid in np.unique(masks)] # 二值化mask
        # ensure input shape is compatible with dinov2
        H, W, _ = rgb.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size
        transformed_rgb = cv2.resize(rgb, (new_W, new_H))
        # 匹配点云和mask的尺寸以及RGB图像的尺寸 和DINOV2的尺寸匹配
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # float32 [H, W, 3]
        # shape info
        shape_info = {
            'img_h': H,
            'img_w': W,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
        return transformed_rgb, rgb, points, masks, shape_info

    # def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat):
    #     projected = rgb.copy()
    #     # overlay keypoints on the image
    #     for keypoint_count, pixel in enumerate(candidate_pixels):
    #         displayed_text = f"{keypoint_count}"
    #         text_length = len(displayed_text)
    #         # draw a box 调整框的大小
    #         box_width = 10 + 10 * (text_length - 1)
    #         box_height = 10
    #         cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
    #         cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
    #         # draw text
    #         org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
    #         color = (255, 0, 0)
    #         font_scale = 0.3
    #         thickness = 1
    #         box_width = int(10 * (font_scale / 0.7)) + 10 * (text_length - 1)
    #         box_height = int(10 * (font_scale / 0.7))
    #         cv2.putText(projected, 
    #                     str(keypoint_count), 
    #                     org, 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 
    #                     font_scale, 
    #                     color, 
    #                     thickness)
    #         keypoint_count += 1
    #     return projected

    import cv2

    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat):
        """
        在图像上以矩形框注释 keypoint 索引数字，并使数字大致居中。
        - rgb: 原图像 (H, W, 3)
        - candidate_pixels: 关键点像素坐标列表，列表中每个元素为 (y, x)
        - 其余参数仅示例保留，不参与演示。
        """
        # 复制图像，避免直接在原图像上修改
        projected = rgb.copy()

        # 遍历每个关键点像素
        for keypoint_count, pixel in enumerate(candidate_pixels):
            # 1) 要显示的数字（文本）
            displayed_text = str(keypoint_count)

            # 2) 计算文字尺寸 (width, height)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.2  # 控制字体大小，可根据需要调整
            text_thickness = 1  # 文字线条粗细
            text_size = cv2.getTextSize(displayed_text, font, font_scale, text_thickness)[0]

            # 3) 计算矩形框的大小
            #    给文字预留一些上下左右边距，以免文字贴到边框
            padding = 1
            box_width = text_size[0] + 2 * padding
            box_height = text_size[1] + 2 * padding

            # 4) 根据中心 (pixel[1], pixel[0]) 计算出矩形的左上角和右下角坐标
            center_x, center_y = pixel[1], pixel[0]
            top_left = (center_x - box_width // 2, center_y - box_height // 2)
            bottom_right = (center_x + box_width // 2, center_y + box_height // 2)

            # 5) 先画一个填充的矩形（白色背景）
            cv2.rectangle(
                projected,
                top_left,
                bottom_right,
                (255, 255, 255),  # 白色填充
                -1  # thickness = -1 表示填充
            )

            # 6) 再画一个边框（黑色），可以调整边框粗细
            box_thickness = 0
            cv2.rectangle(
                projected,
                top_left,
                bottom_right,
                (255, 255, 255),  # 黑色边框
                box_thickness
            )

            # 7) 计算文字放置点，让文字尽量居中
            #    OpenCV 的坐标系：x 对应列方向，y 对应行方向，(x, y) = (列, 行)
            text_x = center_x - text_size[0] // 2
            # 文字基线在字体底部，因此 y 坐标要往下移动半个文字高度
            text_y = center_y + text_size[1] // 2

            # 8) 在矩形框中心绘制文字 (红色)
            cv2.putText(
                projected,
                displayed_text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 255),  # 红色字体
                text_thickness,
                cv2.LINE_AA
            )

        return projected

    def register_keypoints(self, keypoints, points, mask):
        """
        使用图像点云和 mask 数据处理关键点，并将其映射到真实像素位置和物体。

        Args:
            keypoints (np.ndarray): 初始关键点的 3D 坐标数组，形状为 (N, 3)。
            points (np.ndarray): 点云数据，形状为 (H, W, 3)，表示图像中每个像素对应的 3D 点。
            mask (np.ndarray): 分割掩码，形状为 (H, W)，值为 0 表示背景，正整数表示不同物体。

        Returns:
            tuple: (updated_keypoints, keypoint_info)
                - updated_keypoints (np.ndarray): 更新后的关键点 3D 坐标数组，形状为 (N, 3)。
                - keypoint_info (list): 每个关键点的附加信息列表，包括：
                    - 物体类别 ID (mask 值)。
                    - 对应像素坐标 (x, y)。
        """
        updated_keypoints = []  # 存储更新后的关键点
        keypoint_info = []  # 存储关键点的附加信息

        H, W, _ = points.shape

        for keypoint in keypoints:
            closest_distance = float('inf')
            updated_position = keypoint
            pixel_coords = None
            object_id = None

            # 遍历掩码中每个物体的前景像素点
            for y in range(H):
                for x in range(W):
                    if mask[y, x] == 0:
                        continue

                    # 获取当前像素的 3D 点坐标
                    point = points[y, x]
                    distance = np.linalg.norm(point - keypoint)

                    if distance < closest_distance:
                        closest_distance = distance
                        updated_position = point
                        pixel_coords = (x, y)
                        object_id = mask[y, x]

            updated_keypoints.append(updated_position)
            keypoint_info.append({
                "frame": 0,
                "object_id": object_id,
                "pixel_coords": pixel_coords
            })

        return np.array(updated_keypoints), keypoint_info

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):  # 为每个像素点提取特征
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']
        # get features
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(
            self.device)  # float32 [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        features_dict = self.dinov2.forward_features(img_tensors)
        raw_feature_grid = features_dict['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w,
                                                    -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
        # compute per-point feature using bilinear interpolation
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),
                                                # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(img_h, img_w),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(
            0)  # float32 [H, W, feature_dim]
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[
            -1])  # float32 [H*W, feature_dim]
        return features_flat

    def _cluster_features(self, points, features_flat, masks):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []

        num_arrays = len(masks)
        print(f"Number of arrays in masks: {num_arrays}")

        for rigid_group_id, binary_mask in enumerate(masks):
            # ignore mask that is too large
            print(f"rigid_group_id: {rigid_group_id}")
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue
            # consider only foreground features
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            print(f"feature_pixels shape: {feature_pixels.shape}")
            # (f"feature_pixels: {feature_pixels}")

            feature_points = points[binary_mask]
            # reduce dimensionality to be less sensitive to noise and texture
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])
            X = features_pca
            # add feature_pixels as extra dimensions
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch = (feature_points_torch - feature_points_torch.min(0)[0]) / (
                        feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([X, feature_points_torch], dim=-1)  # 特征和点云坐标拼接


            try:
                cluster_ids_x, cluster_centers = kmeans(
                    X=X,
                    num_clusters=self.config['num_candidates_per_mask'],
                    distance='euclidean',
                    device=self.device,
                    # max_iter=100  # <--- 指定
                )
            except Exception as e:
                print(f"K-means 执行失败，异常 {e}。跳过这个掩模。")
                continue

            if torch.isnan(cluster_centers).any():
                print(f"K-means 收敛失败或出现 NaN，跳过这个掩模 (mask id={rigid_group_id}).")
                continue

            cluster_centers = cluster_centers.to(self.device)
            for cluster_id in range(self.config['num_candidates_per_mask']):
                cluster_center = cluster_centers[cluster_id][:3]
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_

        merged_indices = []

        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))

        return merged_indices
