import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class PlotHelper:
    def __init__(self, 
                 pk_chain_left, 
                 pk_chain_right, 
                 T_base_to_fl, 
                 T_base_to_fr, 
                 base_pose_matrix,
                 device='cpu',
                 dtype=torch.float32,
                 info=None,  # 包含相机参数和 RGB 图像的信息字典
                ):
        """
        初始化 PlotHelper 类。
        """
        self.pk_chain_left = pk_chain_left
        self.pk_chain_right = pk_chain_right
        self.T_base_to_fl = T_base_to_fl
        self.T_base_to_fr = T_base_to_fr
        self.base_pose_matrix = base_pose_matrix
        self.device = device
        self.dtype = dtype
        self._buffered_data = {}  # 用于存储 {count: {current_t: (traj, pred_x0, new_traj)}}
        self._last_count = None   # 记录上一次使用的 count

        if info is None:
            raise ValueError("Info dictionary must be provided.")
        self.info = info

        # 提取 head_camera 的参数
        head_camera_info = self.info.get("observation", {}).get("head_camera", {})
        if not head_camera_info:
            raise ValueError("Head camera information is missing in info dictionary.")

        # 提取相机内参、外参和 RGB 图像
        self.camera_intrinsic_cv = head_camera_info.get("intrinsic_cv")
        self.camera_extrinsic_cv = head_camera_info.get("extrinsic_cv")
        self.cam2world_gl = head_camera_info.get("cam2world_gl")
        self.rgb_image = head_camera_info.get("rgb")

        # if self.camera_intrinsic_cv is None or self.camera_extrinsic_cv is None or self.cam2world_gl is None:
        #     raise ValueError("Camera intrinsic, extrinsic or cam2world_gl information is missing.")

        # if self.rgb_image is None:
        #     raise ValueError("RGB image is missing in head_camera information.")

        # 计算相机到世界的变换矩阵
        if self.camera_extrinsic_cv.shape == (3, 4):
            self.camera_extrinsic_cv = np.vstack([self.camera_extrinsic_cv, [0, 0, 0, 1]])  # 转为 4x4
            self.world_to_camera = np.linalg.inv(self.camera_extrinsic_cv)
        elif self.camera_extrinsic_cv.shape == (4, 4):
            self.world_to_camera = np.linalg.inv(self.camera_extrinsic_cv)

    def pose7_to_mat4x4(self, pose7):
        """
        将7维位姿转换为4x4变换矩阵。

        参数:
            pose7 (list or np.ndarray): [x, y, z, qx, qy, qz, qw]

        返回:
            torch.Tensor: 4x4 变换矩阵
        """
        x, y, z, qx, qy, qz, qw = pose7
        mat = torch.eye(4, device=self.device, dtype=self.dtype)
        mat[0, 3] = x
        mat[1, 3] = y
        mat[2, 3] = z
        # 将四元数转换为旋转矩阵
        q = torch.tensor([qw, qx, qy, qz], device=self.device, dtype=self.dtype)
        rot = self.quaternion_to_rotation_matrix(q)
        mat[:3, :3] = rot
        return mat

    def quaternion_to_rotation_matrix(self, q):
        """
        将四元数转换为旋转矩阵。

        参数:
            q (torch.Tensor): 四元数 [qw, qx, qy, qz]

        返回:
            torch.Tensor: 3x3 旋转矩阵
        """
        qw, qx, qy, qz = q
        rot = torch.tensor([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ], device=self.device, dtype=self.dtype)
        return rot

    def convert_actions_to_world(self, actions):
        """
        将动作转换为世界坐标系中的末端执行器位置。

        参数:
            actions (torch.Tensor): 形状为 (1, 16, 14) 的动作张量

        返回:
            dict: {'left': (16, 4, 4), 'right': (16, 4, 4)}
        """
        # 假设 actions 包含两个臂的关节角度
        # 分离左臂和右臂的关节角度
        left_actions = actions[..., :6].reshape(-1, 6)   # (16, 6)
        right_actions = actions[..., 7:13].reshape(-1, 6)  # (16, 6)

        # ========== 计算左臂末端位姿 (base_link->end_effector) ==========
        all_poses_left = self.pk_chain_left.forward_kinematics(left_actions, end_only=False)
        pose_mats_left = all_poses_left["fl_link6"].get_matrix()  # shape [16, 4, 4]

        # ========== 计算右臂末端位姿 (base_link->end_effector) ==========
        all_poses_right = self.pk_chain_right.forward_kinematics(right_actions, end_only=False)
        pose_mats_right = all_poses_right["fr_link6"].get_matrix()  # shape [16, 4, 4]

        # ========== 若有 base_pose_matrix，则应用 (world->base_link) ==========
        # 假设 base_pose_matrix 形状 [4,4]，表示 world->base_link
        if self.base_pose_matrix is not None:
            base_pose_torch = self.base_pose_matrix
            base_pose_torch = base_pose_torch.unsqueeze(0).expand(pose_mats_left.shape[0], -1, -1)
            # 将 base_link->end_effector 转到 world->end_effector
            pose_mats_left = base_pose_torch @ pose_mats_left  
            pose_mats_right = base_pose_torch @ pose_mats_right  

        return {
            'left': pose_mats_left.cpu().numpy(),   # (16, 4, 4)
            'right': pose_mats_right.cpu().numpy()
        }


    def transform_world_to_camera(self, world_points):
        """
        将世界坐标系中的点转换到相机坐标系。

        参数:
            world_points (np.ndarray): 形状为 (N, 3) 的点云

        返回:
            np.ndarray: 形状为 (N, 3) 的相机坐标系点
        """
        # 转换为齐次坐标
        N = world_points.shape[0]
        homogeneous_world = np.hstack([world_points, np.ones((N, 1))])  # (N, 4)
        # 使用 extrinsic_cv 作为 world_to_camera
        camera_extrinsics = self.world_to_camera  # (4, 4)
        camera_points_hom = (camera_extrinsics @ homogeneous_world.T).T  # (N, 4)
        camera_points = camera_points_hom[:, :3] / camera_points_hom[:, 3:4]
        return camera_points

    def project_to_image(self, camera_points):
        """
        将相机坐标系中的点投影到图像平面。

        参数:
            camera_points (np.ndarray): 形状为 (N, 3) 的相机坐标系点

        返回:
            np.ndarray: 形状为 (N, 2) 的图像坐标
        """
        if self.camera_intrinsic_cv is None:
            raise ValueError("Camera intrinsics not provided.")
        fx = self.camera_intrinsic_cv[0, 0]
        fy = self.camera_intrinsic_cv[1, 1]
        cx = self.camera_intrinsic_cv[0, 2]
        cy = self.camera_intrinsic_cv[1, 2]

        X = camera_points[:, 0]
        Y = camera_points[:, 1]
        Z = camera_points[:, 2]

        # 避免除以零
        Z = np.where(Z == 0, 1e-6, Z)

        u = (fx * X) / Z + cx
        v = (fy * Y) / Z + cy

        return np.vstack([u, v]).T  # (N,2)

    def get_trajectory_world(self, actions):
        """
        获取双臂在世界坐标系中的末端执行器轨迹。

        参数:
            actions (torch.Tensor): 形状为 (1, 16, 14) 的动作张量

        返回:
            dict: {'left': (16, 3), 'right': (16, 3)}
        """
        poses = self.convert_actions_to_world(actions)  # {'left': [16,4,4], 'right': [16,4,4]}
        left_positions = poses['left'][:, :3, 3]  # (16,3)
        right_positions = poses['right'][:, :3, 3]  # (16,3)
        return {
            'left': left_positions,
            'right': right_positions
        }

    def plot_in_world(self, 
                      trajectory_world, 
                      new_trajectory_world, 
                      pred_x0_world=None,
                      title='Trajectory in World Coordinates',
                      save_path=None,
                      show_fig=False):
        """
        在世界坐标系中绘制轨迹，并可选择保存图片。

        参数:
            trajectory_world (dict): {'left': (16,3), 'right': (16,3)}
            new_trajectory_world (dict): {'left': (16,3), 'right': (16,3)}
            title (str): 图标题
            save_path (str): 若不为 None，则将图片保存到该路径
            show_fig (bool): 是否显示图像
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制原始轨迹
        ax.plot(trajectory_world['left'][:,0],
                trajectory_world['left'][:,1],
                trajectory_world['left'][:,2],
                label='Left Arm Original',
                color='blue')
        ax.plot(trajectory_world['right'][:,0],
                trajectory_world['right'][:,1],
                trajectory_world['right'][:,2],
                label='Right Arm Original',
                color='green')
        
        # 绘制新轨迹
        ax.plot(new_trajectory_world['left'][:,0],
                new_trajectory_world['left'][:,1],
                new_trajectory_world['left'][:,2],
                label='Left Arm New',
                color='cyan',
                linestyle='--')
        ax.plot(new_trajectory_world['right'][:,0],
                new_trajectory_world['right'][:,1],
                new_trajectory_world['right'][:,2],
                label='Right Arm New',
                color='lime',
                linestyle='--')
        
        # 如果传入了 pred_x0_world，就在同一图里绘制
        if pred_x0_world is not None:
            ax.plot(pred_x0_world['left'][:,0],
                    pred_x0_world['left'][:,1],
                    pred_x0_world['left'][:,2],
                    label='Left Arm Pred_x0',
                    color='red',
                    linestyle=':')
            ax.plot(pred_x0_world['right'][:,0],
                    pred_x0_world['right'][:,1],
                    pred_x0_world['right'][:,2],
                    label='Right Arm Pred_x0',
                    color='magenta',
                    linestyle=':')
        
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
            print(f"Figure saved to: {save_path}")

        if show_fig:
            plt.show()
        else:
            plt.close(fig)

    def plot_on_rgb(self, 
                    trajectory_world, 
                    new_trajectory_world, 
                    title='Trajectory on RGB Image',
                    save_path=None,
                    show_fig=False):
        """
        在RGB图像上绘制轨迹，并可选择保存图片。

        参数:
            trajectory_world (dict): {'left': (16,3), 'right': (16,3)}
            new_trajectory_world (dict): {'left': (16,3), 'right': (16,3)}
            title (str): 图标题
            save_path (str): 若不为 None，则将图片保存到该路径
            show_fig (bool): 是否显示图像
        """
        fig, ax = plt.subplots()
        ax.imshow(self.rgb_image)
        
        # 将世界坐标转换到相机坐标，再投影到图像
        for arm, color_orig, color_new, marker_orig, marker_new in [
            ('left', 'blue', 'cyan', 'o', 'x'),
            ('right', 'green', 'lime', 'o', 'x')]:

            # 原始轨迹
            camera_points = self.transform_world_to_camera(trajectory_world[arm])  # (16,3)
            image_points = self.project_to_image(camera_points)  # (16,2)

            ax.plot(image_points[:,0],
                    image_points[:,1],
                    label=f'{arm.capitalize()} Original',
                    color=color_orig,
                    marker=marker_orig)

            # 新轨迹
            camera_points_new = self.transform_world_to_camera(new_trajectory_world[arm])  # (16,3)
            image_points_new = self.project_to_image(camera_points_new)  # (16,2)
            ax.plot(image_points_new[:,0],
                    image_points_new[:,1],
                    label=f'{arm.capitalize()} New',
                    color=color_new,
                    marker=marker_new)
        
        ax.set_title(title)
        ax.legend()

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
            print(f"Figure saved to: {save_path}")

        if show_fig:
            plt.show()
        else:
            plt.close(fig)

    # def _flush_plots_for_count(self, old_count):
    #     """
    #     当 count 变更时触发，针对 old_count 做两类图：
    #     1) 将所有 current_t 的 new_trajectory 画在**同一张**图上，但左右手分两个 subplot
    #     2) 对每个 current_t，单独画一张对比 (trajectory vs new_trajectory)，同样左右手分两个 subplot
    #     目标点 (left_point, right_point) 都用星型画在对应 subplot 中。
    #     """
    #     if old_count not in self._buffered_data:
    #         return
        
    #     save_dir = f"/home/pine/RoboTwin/keguide/plot_results/{old_count}"
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir, exist_ok=True)

    #     data_for_count = self._buffered_data[old_count]
    #     # data_for_count: { current_t: (trajectory, pred_x0, new_trajectory, left_point, right_point) }

    #     # 将其中的 (current_t, ...) 按升序或降序排列
    #     all_items = sorted(data_for_count.items(), key=lambda x: x[0])
    #     colors = plt.cm.rainbow(np.linspace(0, 1, len(all_items)))

    #     # ========== 1) 第一类图: 所有 new_trajectory 画在同一张图(左右手分两个 subplot) ==========
    #     fig_all = plt.figure(figsize=(12, 5))
    #     ax_all_left = fig_all.add_subplot(1, 2, 1, projection='3d')
    #     ax_all_right = fig_all.add_subplot(1, 2, 2, projection='3d')

    #     # 取出第一个 item 的 left_point、right_point 作为该 count 的目标点（假设不变）
    #     if len(all_items) > 0:
    #         _, first_data = all_items[0]
    #         _, _, _, left_pt, right_pt = first_data
    #     else:
    #         left_pt, right_pt = None, None

    #     # 目标点(左手)画在左 subplot
    #     if left_pt is not None:
    #         if isinstance(left_pt, torch.Tensor):
    #             left_pt = left_pt.detach().cpu().numpy()
    #         ax_all_left.scatter(left_pt[0], left_pt[1], left_pt[2],
    #                             marker='*', color='black', s=150,
    #                             label='Left Target')

    #     # 目标点(右手)画在右 subplot
    #     if right_pt is not None:
    #         if isinstance(right_pt, torch.Tensor):
    #             right_pt = right_pt.detach().cpu().numpy()
    #         ax_all_right.scatter(right_pt[0], right_pt[1], right_pt[2],
    #                             marker='*', color='red', s=150,
    #                             label='Right Target')

    #     # 把所有 new_trajectory 依次画到这两个 subplot
    #     for color, (current_t, (traj, pred, new_traj, lpt, rpt)) in zip(colors, all_items):
    #         new_world = self.get_trajectory_world(new_traj)
    #         # 左手散点
    #         ax_all_left.scatter(new_world['left'][:,0],
    #                             new_world['left'][:,1],
    #                             new_world['left'][:,2],
    #                             c=[color], marker='^', s=25,
    #                             label=f"Left new (t={current_t})")
    #         # 右手散点
    #         ax_all_right.scatter(new_world['right'][:,0],
    #                             new_world['right'][:,1],
    #                             new_world['right'][:,2],
    #                             c=[color], marker='o', s=25,
    #                             label=f"Right new (t={current_t})")

    #     ax_all_left.set_title(f"[Count={old_count}] All new_trajectories - Left")
    #     ax_all_left.set_xlabel('X')
    #     ax_all_left.set_ylabel('Y')
    #     ax_all_left.set_zlabel('Z')
    #     ax_all_left.legend()

    #     ax_all_right.set_title(f"[Count={old_count}] All new_trajectories - Right")
    #     ax_all_right.set_xlabel('X')
    #     ax_all_right.set_ylabel('Y')
    #     ax_all_right.set_zlabel('Z')
    #     ax_all_right.legend()

    #     # plt.show()  # 不保存，直接调试
    #     plt.savefig(f"{save_dir}/count_{old_count}_all_new.png", dpi=200)
    #     plt.close(fig_all)

    #     # ========== 2) 第二类图: 对每个 current_t，单独画"原始 vs. 新"的两张 subplot(左手/右手) ==========
        
    #     for (current_t, (traj, pred, new_traj, lpt, rpt)) in all_items:
    #         fig_one = plt.figure(figsize=(12, 5))
    #         ax_left = fig_one.add_subplot(1, 2, 1, projection='3d')
    #         ax_right = fig_one.add_subplot(1, 2, 2, projection='3d')

    #         # 若有目标点, 左手画在左 subplot, 右手画在右 subplot
    #         if lpt is not None:
    #             if isinstance(lpt, torch.Tensor):
    #                 lpt = lpt.detach().cpu().numpy()
    #             ax_left.scatter(lpt[0], lpt[1], lpt[2],
    #                             marker='*', color='black', s=150,
    #                             label='Left Target')
    #         if rpt is not None:
    #             if isinstance(rpt, torch.Tensor):
    #                 rpt = rpt.detach().cpu().numpy()
    #             ax_right.scatter(rpt[0], rpt[1], rpt[2],
    #                             marker='*', color='red', s=150,
    #                             label='Right Target')

    #         traj_world = self.get_trajectory_world(traj)
    #         new_traj_world = self.get_trajectory_world(new_traj)

    #         # ---- 左手：原始(蓝色圆) vs 新(绿色方块) ----
    #         ax_left.scatter(traj_world['left'][:,0],
    #                         traj_world['left'][:,1],
    #                         traj_world['left'][:,2],
    #                         c='blue', marker='o', s=20,
    #                         label='Left Orig')
    #         ax_left.scatter(new_traj_world['left'][:,0],
    #                         new_traj_world['left'][:,1],
    #                         new_traj_world['left'][:,2],
    #                         c='green', marker='s', s=20,
    #                         label='Left New')
    #         ax_left.set_title(f"[Count={old_count}, t={current_t}] Left")
    #         ax_left.set_xlabel('X')
    #         ax_left.set_ylabel('Y')
    #         ax_left.set_zlabel('Z')
    #         ax_left.legend()

    #         # ---- 右手：原始(蓝色三角) vs 新(绿色菱形) ----
    #         ax_right.scatter(traj_world['right'][:,0],
    #                         traj_world['right'][:,1],
    #                         traj_world['right'][:,2],
    #                         c='blue', marker='^', s=20,
    #                         label='Right Orig')
    #         ax_right.scatter(new_traj_world['right'][:,0],
    #                         new_traj_world['right'][:,1],
    #                         new_traj_world['right'][:,2],
    #                         c='green', marker='D', s=20,
    #                         label='Right New')
    #         ax_right.set_title(f"[Count={old_count}, t={current_t}] Right")
    #         ax_right.set_xlabel('X')
    #         ax_right.set_ylabel('Y')
    #         ax_right.set_zlabel('Z')
    #         ax_right.legend()

    #         # plt.show()  # 显示，方便调试
    #         plt.savefig(f"{save_dir}/count_{old_count}_t_{current_t}.png", dpi=200)
    #         plt.close(fig_one)

    #     # 最后清空缓存
    #     del self._buffered_data[old_count]

    def _flush_plots_for_count(self, old_count):
        """
        当 count 变更时触发，针对 old_count 做两类图：
        1) 将所有 current_t 的 new_trajectory 画在同一张图，但在 XY / XZ / YZ 平面投影上分别显示，
            左右手分两个列 => 总共 3行×2列 = 6 个子图。
        2) 对每个 current_t，单独画一张对比图 (trajectory vs new_trajectory)，同样在 XY / XZ / YZ 上投影，
            左右手分两个列 => 同样 3行×2列 = 6 个子图。
        目标点 (left_point, right_point) 投影到对应平面上，用星型标记。
        """

        if old_count not in self._buffered_data:
            return

        # save_dir = f"/home/pine/RoboTwin/keguide/plot_results/{old_count}"
        save_dir = f"/data1/hydeng/Keguide_RoboTwin/keguide/plot_results/{old_count}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        data_for_count = self._buffered_data[old_count]
        # data_for_count: { current_t: (trajectory, pred_x0, new_trajectory, left_point, right_point) }

        # 将 (current_t, ...) 按升序或降序排列
        all_items = sorted(data_for_count.items(), key=lambda x: x[0])
        colors = plt.cm.rainbow(np.linspace(0, 1, len(all_items)))

        #=====================#
        #  Helper 函数: 取投影
        #=====================#
        def get_xy(p):  # p是一个(N,3)或(3,)的numpy数组
            return p[...,0], p[...,1]  # (x, y)
        def get_xz(p):
            return p[...,0], p[...,2]  # (x, z)
        def get_yz(p):
            return p[...,1], p[...,2]  # (y, z)

        #===================================================================#
        # 1) 第一类图: 所有 new_trajectory 画在同一张图(3行×2列: XY/XZ/YZ × (Left/Right))
        #===================================================================#
        fig_all = plt.figure(figsize=(12, 12))

        # 约定：第一列画左手(3个平面)，第二列画右手(3个平面)
        # row=1 -> XY, row=2 -> XZ, row=3 -> YZ
        ax_left_xy = fig_all.add_subplot(3, 2, 1)
        ax_right_xy= fig_all.add_subplot(3, 2, 2)
        ax_left_xz = fig_all.add_subplot(3, 2, 3)
        ax_right_xz= fig_all.add_subplot(3, 2, 4)
        ax_left_yz = fig_all.add_subplot(3, 2, 5)
        ax_right_yz= fig_all.add_subplot(3, 2, 6)

        # 取出第一个 item 的目标点
        if len(all_items) > 0:
            _, first_data = all_items[0]
            _, _, _, left_pt, right_pt = first_data
        else:
            left_pt, right_pt = None, None

        # 将目标点投影到三个平面中，分别在(左/右)手对应的子图上画星型
        if left_pt is not None:
            if isinstance(left_pt, torch.Tensor):
                left_pt = left_pt.detach().cpu().numpy()
            # left_pt.shape=(3,)
            # xy
            x, y = get_xy(left_pt)
            ax_left_xy.scatter(x, y, marker='*', color='black', s=150, label='Left Target')
            # xz
            x, z = get_xz(left_pt)
            ax_left_xz.scatter(x, z, marker='*', color='black', s=150, label='Left Target')
            # yz
            y, z = get_yz(left_pt)
            ax_left_yz.scatter(y, z, marker='*', color='black', s=150, label='Left Target')

        if right_pt is not None:
            if isinstance(right_pt, torch.Tensor):
                right_pt = right_pt.detach().cpu().numpy()
            # xy
            x, y = get_xy(right_pt)
            ax_right_xy.scatter(x, y, marker='*', color='red', s=150, label='Right Target')
            # xz
            x, z = get_xz(right_pt)
            ax_right_xz.scatter(x, z, marker='*', color='red', s=150, label='Right Target')
            # yz
            y, z = get_yz(right_pt)
            ax_right_yz.scatter(y, z, marker='*', color='red', s=150, label='Right Target')

        # 依次画所有 new_trajectory 的投影
        for color, (current_t, (traj, pred, new_traj, lpt, rpt)) in zip(colors, all_items):
            new_world = self.get_trajectory_world(new_traj)  # {'left': (N,3), 'right': (N,3)}

            # 左手
            left_xyz = new_world['left']  # shape=(N,3)
            lx_xy, ly_xy = get_xy(left_xyz)  # XY投影
            lx_xz, lz_xz = get_xz(left_xyz)  # XZ投影
            ly_yz, lz_yz = get_yz(left_xyz)  # YZ投影

            ax_left_xy.scatter(lx_xy, ly_xy, c=[color], marker='^', s=25, label=f"Left(t={current_t})")
            ax_left_xz.scatter(lx_xz, lz_xz, c=[color], marker='^', s=25, label=f"Left(t={current_t})")
            ax_left_yz.scatter(ly_yz, lz_yz, c=[color], marker='^', s=25, label=f"Left(t={current_t})")

            # 右手
            right_xyz = new_world['right']
            rx_xy, ry_xy = get_xy(right_xyz)
            rx_xz, rz_xz = get_xz(right_xyz)
            ry_yz, rz_yz = get_yz(right_xyz)

            ax_right_xy.scatter(rx_xy, ry_xy, c=[color], marker='o', s=25, label=f"Right(t={current_t})")
            ax_right_xz.scatter(rx_xz, rz_xz, c=[color], marker='o', s=25, label=f"Right(t={current_t})")
            ax_right_yz.scatter(ry_yz, rz_yz, c=[color], marker='o', s=25, label=f"Right(t={current_t})")

        # 设置标题、坐标轴标签等
        ax_left_xy.set_title(f"Left XY (count={old_count})")
        ax_left_xy.set_xlabel('X'); ax_left_xy.set_ylabel('Y')
        ax_left_xy.legend()

        ax_right_xy.set_title(f"Right XY (count={old_count})")
        ax_right_xy.set_xlabel('X'); ax_right_xy.set_ylabel('Y')
        ax_right_xy.legend()

        ax_left_xz.set_title(f"Left XZ (count={old_count})")
        ax_left_xz.set_xlabel('X'); ax_left_xz.set_ylabel('Z')
        ax_left_xz.legend()

        ax_right_xz.set_title(f"Right XZ (count={old_count})")
        ax_right_xz.set_xlabel('X'); ax_right_xz.set_ylabel('Z')
        ax_right_xz.legend()

        ax_left_yz.set_title(f"Left YZ (count={old_count})")
        ax_left_yz.set_xlabel('Y'); ax_left_yz.set_ylabel('Z')
        ax_left_yz.legend()

        ax_right_yz.set_title(f"Right YZ (count={old_count})")
        ax_right_yz.set_xlabel('Y'); ax_right_yz.set_ylabel('Z')
        ax_right_yz.legend()

        # 保存并关闭
        plt.savefig(f"{save_dir}/count_{old_count}_all_new.png", dpi=200)
        plt.close(fig_all)

        #===================================================================================#
        # 2) 第二类图: 对每个 current_t，单独画"原始 vs. 新"的投影(3行×2列: XY/XZ/YZ × (Left/Right))
        #===================================================================================#
        for (current_t, (traj, pred, new_traj, lpt, rpt)) in all_items:
            fig_one = plt.figure(figsize=(12, 12))

            # 按照 3行×2列，分别是:
            # (1,1): 左手XY, (1,2): 右手XY
            # (2,1): 左手XZ, (2,2): 右手XZ
            # (3,1): 左手YZ, (3,2): 右手YZ
            ax_l_xy = fig_one.add_subplot(3, 2, 1)
            ax_r_xy = fig_one.add_subplot(3, 2, 2)
            ax_l_xz = fig_one.add_subplot(3, 2, 3)
            ax_r_xz = fig_one.add_subplot(3, 2, 4)
            ax_l_yz = fig_one.add_subplot(3, 2, 5)
            ax_r_yz = fig_one.add_subplot(3, 2, 6)

            # 若有目标点
            if lpt is not None:
                if isinstance(lpt, torch.Tensor):
                    lpt = lpt.detach().cpu().numpy()
                # 分别投影并画在左手的三个子图
                lx_xy, ly_xy = get_xy(lpt)
                lx_xz, lz_xz = get_xz(lpt)
                ly_yz, lz_yz = get_yz(lpt)
                ax_l_xy.scatter(lx_xy, ly_xy, marker='*', color='black', s=150, label='Left Target')
                ax_l_xz.scatter(lx_xz, lz_xz, marker='*', color='black', s=150, label='Left Target')
                ax_l_yz.scatter(ly_yz, lz_yz, marker='*', color='black', s=150, label='Left Target')

            if rpt is not None:
                if isinstance(rpt, torch.Tensor):
                    rpt = rpt.detach().cpu().numpy()
                rx_xy, ry_xy = get_xy(rpt)
                rx_xz, rz_xz = get_xz(rpt)
                ry_yz, rz_yz = get_yz(rpt)
                ax_r_xy.scatter(rx_xy, ry_xy, marker='*', color='red', s=150, label='Right Target')
                ax_r_xz.scatter(rx_xz, rz_xz, marker='*', color='red', s=150, label='Right Target')
                ax_r_yz.scatter(ry_yz, rz_yz, marker='*', color='red', s=150, label='Right Target')

            traj_world     = self.get_trajectory_world(traj)       # 原始
            new_traj_world = self.get_trajectory_world(new_traj)   # 新

            # 左手
            traj_l = traj_world['left']
            newl   = new_traj_world['left']
            # XY
            tlx_xy, tly_xy = get_xy(traj_l)
            nlx_xy, nly_xy = get_xy(newl)
            ax_l_xy.scatter(tlx_xy, tly_xy, c='blue', marker='o', s=20, label='Left Orig')
            ax_l_xy.scatter(nlx_xy, nly_xy, c='green', marker='s', s=20, label='Left New')
            # XZ
            tlx_xz, tlz_xz = get_xz(traj_l)
            nlx_xz, nlz_xz = get_xz(newl)
            ax_l_xz.scatter(tlx_xz, tlz_xz, c='blue', marker='o', s=20, label='Left Orig')
            ax_l_xz.scatter(nlx_xz, nlz_xz, c='green', marker='s', s=20, label='Left New')
            # YZ
            tly_yz, tlz_yz = get_yz(traj_l)
            nly_yz, nlz_yz = get_yz(newl)
            ax_l_yz.scatter(tly_yz, tlz_yz, c='blue', marker='o', s=20, label='Left Orig')
            ax_l_yz.scatter(nly_yz, nlz_yz, c='green', marker='s', s=20, label='Left New')

            # 右手
            traj_r = traj_world['right']
            newr   = new_traj_world['right']
            # XY
            rtx_xy, rty_xy = get_xy(traj_r)
            nrx_xy, nry_xy = get_xy(newr)
            ax_r_xy.scatter(rtx_xy, rty_xy, c='blue', marker='^', s=20, label='Right Orig')
            ax_r_xy.scatter(nrx_xy, nry_xy, c='green', marker='D', s=20, label='Right New')
            # XZ
            rtx_xz, rtz_xz = get_xz(traj_r)
            nrx_xz, nrz_xz = get_xz(newr)
            ax_r_xz.scatter(rtx_xz, rtz_xz, c='blue', marker='^', s=20, label='Right Orig')
            ax_r_xz.scatter(nrx_xz, nrz_xz, c='green', marker='D', s=20, label='Right New')
            # YZ
            rty_yz, rtz_yz = get_yz(traj_r)
            nry_yz, nrz_yz = get_yz(newr)
            ax_r_yz.scatter(rty_yz, rtz_yz, c='blue', marker='^', s=20, label='Right Orig')
            ax_r_yz.scatter(nry_yz, nrz_yz, c='green', marker='D', s=20, label='Right New')

            # 设置标题、坐标、图例
            ax_l_xy.set_title(f"Left XY (count={old_count}, t={current_t})")
            ax_l_xy.set_xlabel('X'); ax_l_xy.set_ylabel('Y'); ax_l_xy.legend()

            ax_r_xy.set_title(f"Right XY (count={old_count}, t={current_t})")
            ax_r_xy.set_xlabel('X'); ax_r_xy.set_ylabel('Y'); ax_r_xy.legend()

            ax_l_xz.set_title(f"Left XZ (count={old_count}, t={current_t})")
            ax_l_xz.set_xlabel('X'); ax_l_xz.set_ylabel('Z'); ax_l_xz.legend()

            ax_r_xz.set_title(f"Right XZ (count={old_count}, t={current_t})")
            ax_r_xz.set_xlabel('X'); ax_r_xz.set_ylabel('Z'); ax_r_xz.legend()

            ax_l_yz.set_title(f"Left YZ (count={old_count}, t={current_t})")
            ax_l_yz.set_xlabel('Y'); ax_l_yz.set_ylabel('Z'); ax_l_yz.legend()

            ax_r_yz.set_title(f"Right YZ (count={old_count}, t={current_t})")
            ax_r_yz.set_xlabel('Y'); ax_r_yz.set_ylabel('Z'); ax_r_yz.legend()

            plt.savefig(f"{save_dir}/count_{old_count}_t_{current_t}.png", dpi=200)
            plt.close(fig_one)

        # 最后清空缓存
        del self._buffered_data[old_count]

    def plot_trajectories(self, 
                          trajectory, 
                          pred_x0, 
                          new_trajectory, 
                          count, 
                          current_t, 
                          left_point, 
                          right_point):
        """
        每次只缓存数据，不立即画图。
        当检测到 count 切换时，先把上一个 count 的数据统一画图并清空。
        """
        # 如果 count 变了，先 flush 上一个 count
        if self._last_count is not None and self._last_count != count:
            self._flush_plots_for_count(self._last_count)

        if self._last_count is None or self._last_count != count:
            self._last_count = count

        # 缓存本次
        if count not in self._buffered_data:
            self._buffered_data[count] = {}
        self._buffered_data[count][current_t] = (
            trajectory, pred_x0, new_trajectory, left_point, right_point
        )