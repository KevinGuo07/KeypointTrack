# keguide/utils/gradient_helper.py

import torch
import pytorch_kinematics as pk
from typing import Optional
import numpy as np
from keguide.utils.plot_helper import PlotHelper

class DiffusionCostGuidance:
    """
    一个简单的可插拔引导类，用于在扩散采样循环中，
    根据 LLM 的决策(或其它逻辑)对 trajectory 施加梯度引导。
    """
    _plot_helper_static = None

    def __init__(
        self,
        time_step_ratio: float = 1,
        guidance_scale: float = 0.6,
        urdf_path: Optional[str] = None,
        base_pose: Optional[list] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        left_root_link: str = "fl_base_link",
        left_end_link: str = "fl_link6",
        right_root_link: str = "fr_base_link",
        right_end_link: str = "fr_link6",
        rgb_image: Optional[dict] = None,
        count: int = 0,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.time_step_ratio = time_step_ratio
        self.guidance_scale = guidance_scale
        self.left_end_link = left_end_link
        self.right_end_link = right_end_link
        self.info = {
            "observation": {
                "head_camera": {}
            }
        }
        self.info["observation"]["head_camera"] = {
            "intrinsic_cv": rgb_image["intrinsic_cv"],
            "extrinsic_cv": rgb_image["extrinsic_cv"],
            "cam2world_gl": rgb_image["cam2world_gl"],
            "rgb": rgb_image["rgb"],
            "fov": rgb_image["fov"],
        }
        # print("intrinsic_cv",self.info["observation"]["head_camera"]["intrinsic_cv"])
        # print("extrinsic_cv",self.info["observation"]["head_camera"]["extrinsic_cv"])
        # print("cam2world_gl",self.info["observation"]["head_camera"]["cam2world_gl"])

        self.T_base_to_fl = torch.tensor([[1, 0, 0, 0.233],
                                        [0, 1, 0, 0.300],
                                        [0, 0, 1, 0.6275],
                                        [0, 0, 0, 1]], dtype=dtype, device=self.device)
        
        self.T_base_to_fr = torch.tensor([[1, 0, 0, 0.233],
                                        [0, 1, 0, -0.300],
                                        [0, 0, 1, 0.6275],
                                        [0, 0, 0, 1]], dtype=dtype, device=self.device)
        
        self.dtype = dtype
        self.pk_chain_left = None
        self.pk_chain_right = None

        if urdf_path is not None:
            with open(urdf_path, "rb") as f:
                urdf_data = f.read()

            # 构建可微FK链，并放到指定device/dtype
            self.pk_chain_left = (
                pk.build_serial_chain_from_urdf(
                    urdf_data, 
                    end_link_name="fl_link6",
                    root_link_name="fl_base_link",
                )
                .to(dtype=self.dtype, device=self.device)
            )
            self.pk_chain_right = (
                pk.build_serial_chain_from_urdf(
                    urdf_data,
                    end_link_name="fr_link6",
                    root_link_name="fr_base_link",
                )
                .to(dtype=self.dtype, device=self.device)
            )

        self.base_pose_matrix = None
        if base_pose is not None and len(base_pose) == 7:
            self.base_pose_matrix = self.pose7_to_mat4x4(base_pose)

        # 如果还没有创建过，就在这里new一次
        if DiffusionCostGuidance._plot_helper_static is None:
            DiffusionCostGuidance._plot_helper_static = PlotHelper(
                pk_chain_left=self.pk_chain_left,
                pk_chain_right=self.pk_chain_right,
                T_base_to_fl=self.T_base_to_fl,
                T_base_to_fr=self.T_base_to_fr,
                base_pose_matrix=self.base_pose_matrix,
                device=self.device,
                dtype=self.dtype,
                info=self.info
            )

        self.plot_helper = DiffusionCostGuidance._plot_helper_static
        # self.plot_helper._last_count = count
        self.plot_helper.rgb_image = self.info["observation"]["head_camera"]["rgb"]

        self.point_left = None
        self.point_right = None

    def pose7_to_mat4x4(self, pose7):
        T = torch.eye(4, dtype=torch.float32, device=self.device)  # 使用 PyTorch 创建单位矩阵，确保在正确的设备上
        T[:3, 3] = torch.tensor(pose7[:3], dtype=torch.float32, device=self.device)  # 设置平移部分
        rot = pk.quaternion_to_matrix(torch.tensor(pose7[3:], dtype=torch.float32, device=self.device))  # 转换四元数为矩阵
        T[:3, :3] = rot  # 设置旋转部分
        return T

    def apply_guidance(
        self,
        xt: torch.Tensor,         # 当前 noisy sample: x_t
        current_t: int,           # 当前时间步
        timesteps: torch.Tensor,  # 扩散调度的时间步列表
        llm_decision=None,
        std_dev_t=None,
        alpha_t=None, 
        et=None,
        count=None,
        **kwargs
    ) -> torch.Tensor:

        idx_list = (timesteps == current_t).nonzero(as_tuple=True)
        if len(idx_list[0]) == 0:
            return xt
        idx = idx_list[0].item()

        threshold_step = int(len(timesteps) * (1.0 - self.time_step_ratio))
        if idx < threshold_step:
            return xt

        if not self._check_affordance_enable(llm_decision):
            return xt
        
        if alpha_t is None:
            print("alpha_t is None, do nothing")
            return xt
        
        with torch.set_grad_enabled(True):
            # 确保 xt 需要梯度
            xt = xt.requires_grad_(True)
            b, t, d = xt.shape

            sqrt_one_minus_at = (1.0 - alpha_t).sqrt()
            sqrt_at = alpha_t.sqrt()

            # x0_t 依赖于 xt
            x0_t = (xt - et * sqrt_one_minus_at) / sqrt_at

            cost = self._compute_cost(x0_t=x0_t, llm_decision=llm_decision)
            if (cost is None) or (cost == 0.0):
                return xt

            print(f"cost at t={current_t} is {cost.item()}")

            # 计算针对 xt 的梯度
            grad = torch.autograd.grad(cost, xt, retain_graph=False, create_graph=False)[0]
            grad_norm = torch.linalg.norm(grad)
            r = torch.sqrt(torch.tensor(7)) * std_dev_t # (std_dev_t if std_dev_t > 0.1 else 0.1)
            grad = (grad / grad_norm) * r # 归一化梯度

        # 用 no_grad() 更新 xt
        with torch.no_grad():
            new_trajectory = xt - self.guidance_scale * grad
            print(f"new_trajectory has updated at t={current_t}")

        print("current_count", count)

        # self.plot_helper.plot_trajectories(
        #     trajectory = xt.detach(), 
        #     pred_x0 = x0_t.detach(), 
        #     new_trajectory = new_trajectory.detach(), 
        #     count = count, 
        #     current_t = current_t,
        #     left_point = self.point_left,
        #     right_point = self.point_right)

        # 返回新的 trajectory
        x0_t = x0_t.detach()
        new_trajectory = new_trajectory.detach()
        return new_trajectory
        
    def _check_affordance_enable(self, llm_decision):
        if not llm_decision:
            return False
        aff = llm_decision.get("enable_affordance_guidance", {})
        return aff.get("enable", False)

    def _compute_cost(self, x0_t: torch.Tensor, llm_decision=None):
        """
        计算左右臂末端 (在世界坐标系下) 与目标点的距离 cost。
        使用:
        - self.pk_chain_left, self.pk_chain_right: 左右臂的 pk chain
        - self.T_base_to_fl, self.T_base_to_fr: (base_link->fl_base_link / fr_base_link)
        - self.base_pose_matrix: (world->base_link)
        """

        if not llm_decision:
            return None
        aff = llm_decision.get("enable_affordance_guidance", {})
        if not aff.get("enable", False):
            return None

        B, T, D = x0_t.shape
        cost = 0.0

        # ========== 左臂 =============
        left_conf = aff.get("enable_left_arm", {})
        if left_conf.get("enable", False) and (self.pk_chain_left is not None):
            # 1) 取左臂关节 => shape [B*T, 6]
            left_arm_joints = x0_t[..., :6].reshape(-1, 6).to(self.device, self.dtype)

            # 2) 计算 fl_base_link->end_effector 末端坐标相对于左臂base_link的变换
            all_poses_left = self.pk_chain_left.forward_kinematics(left_arm_joints, end_only=False)
            pose_mats_left = all_poses_left["fl_link6"].get_matrix()  # shape [N,4,4]

            # 3) 若有 base_pose_matrix: [world->base_link]
            if self.base_pose_matrix is not None:
                base_pose_torch = self.base_pose_matrix  # 直接使用，无需转换
                base_pose_torch = base_pose_torch.unsqueeze(0).expand(pose_mats_left.shape[0], -1, -1)
                pose_mats_left = base_pose_torch @ pose_mats_left  # => world->end_effector
            
            # 4）末端方向延伸家爪长度0.12
            offset_distance = 0.12
            R = pose_mats_left[:, :3, :3]
            Tran = pose_mats_left[:, :3, 3]
            z_axis = R[:, :, 2]
            t_extend = Tran + z_axis * offset_distance
            pose_mats_left[:, :3, 3] = t_extend

            # 5) 提取末端 xyz，计算距离
            xyz_left = pose_mats_left[:, :3, 3].reshape(B, T, 3)
            xy_left = xyz_left[..., :2]
            self.point_left = torch.tensor(left_conf["point"], device=self.device, dtype=self.dtype)
            self.point_left_2d = torch.tensor(left_conf["point"][:2], device=self.device, dtype=self.dtype)
            diff_left = xyz_left - self.point_left  # broadcast
            # diff_left = xy_left - self.point_left_2d
            cost_left = diff_left.norm(dim=-1).mean()  # L2范数 => scalar
            cost += cost_left

        # ========== 右臂 =============
        right_conf = aff.get("enable_right_arm", {})
        if right_conf.get("enable", False) and (self.pk_chain_right is not None):
            # 1) 取右臂关节 => shape [B*T, 6]
            right_arm_joints = x0_t[..., 7:13].reshape(-1, 6).to(self.device, self.dtype)

            # 2) fr_base_link->end_effector
            all_poses_right = self.pk_chain_right.forward_kinematics(right_arm_joints, end_only=False)
            pose_mats_r = all_poses_right["fr_link6"].get_matrix()  # shape [N,4,4]

            # 3) [world->base_link]
            if self.base_pose_matrix is not None:
                base_pose_torch = self.base_pose_matrix  # 直接使用，无需转换
                base_pose_torch = base_pose_torch.unsqueeze(0).expand(pose_mats_r.shape[0], -1, -1)
                pose_mats_r = base_pose_torch @ pose_mats_r  # => world->end_effector

            # 4) 末端方向延伸家爪长度0.12
            offset_distance = 0.12
            R_1 = pose_mats_r[:, :3, :3]
            Tran_1 = pose_mats_r[:, :3, 3]
            z_axis_1 = R_1[:, :, 2]
            t_extend_1 = Tran_1 + z_axis_1 * offset_distance
            pose_mats_r[:, :3, 3] = t_extend_1

            # 5) xyz
            xyz_r = pose_mats_r[:, :3, 3].reshape(B, T, 3)
            xy_right = xyz_r[..., :2]
            self.point_right = torch.tensor(right_conf["point"], device=self.device, dtype=self.dtype)
            self.point_right_2d = torch.tensor(right_conf["point"][:2], device=self.device, dtype=self.dtype)
            diff_right = xyz_r - self.point_right
            # diff_right = xy_right - self.point_right_2d
            cost_right = diff_right.norm(dim=-1).mean()
            cost += cost_right

        return cost if cost != 0 else None

