import sapien
import torch
import pytorch_kinematics as pk
from sapien.render import set_global_config
import mplib.planner
import mplib
import numpy as np  

# 确保PyTorch已安装并且CUDA可用
if torch.cuda.is_available():
    device = torch.device("cuda")  # 或者指定具体的GPU，如 "cuda:0"
else:
    device = torch.device("cpu")  # 如果没有GPU，回退到CPU

def endpose_transform(joint, gripper_val):
        rpy = joint.global_pose.get_rpy()
        roll, pitch, yaw = rpy
        x,y,z = joint.global_pose.p
        endpose = {
            "gripper": float(gripper_val),
            "pitch" : float(pitch),
            "roll" : float(roll),
            "x": float(x),
            "y": float(y),
            "yaw" : float(yaw),
            "z": float(z),
        }
        return endpose

base_pose = [0, -0.65, 0, 1, 0, 0, 1]

def pose7_to_mat4x4(pose7):
    T = torch.eye(4, dtype=torch.float32, device=device)  # 使用 PyTorch 创建单位矩阵，确保在正确的设备上
    T[:3, 3] = torch.tensor(pose7[:3], dtype=torch.float32, device=device)  # 设置平移部分
    rot = pk.quaternion_to_matrix(torch.tensor(pose7[3:], dtype=torch.float32, device=device))  # 转换四元数为矩阵
    T[:3, :3] = rot  # 设置旋转部分
    return T

def mat4x4_to_pose7(T):
    pose7 = torch.zeros(7, dtype=torch.float32, device=device)  # 在 GPU 上创建 tensor
    pose7[:3] = T[:3, 3]  # 提取平移部分
    pose7[3:] = pk.matrix_to_quaternion(T[:3, :3])  # 将旋转矩阵转换回四元数
    return pose7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_pose_matrix = pose7_to_mat4x4(base_pose)


urdf_path = f"/home/pine/RoboTwin/aloha_maniskill_sim/urdf/arx5_description_isaac.urdf"
srdf_path = f"/home/pine/RoboTwin/aloha_maniskill_sim/srdf/arx5_description_isaac.srdf"

with open(urdf_path, "rb") as f:
    urdf_data = f.read()

# pytorch kinematics
pk_chain_left = (
                pk.build_serial_chain_from_urdf(
                    urdf_data, 
                    end_link_name="fl_link6",
                    root_link_name="fl_base_link",
                ))

pk_chain_right = (
                pk.build_serial_chain_from_urdf(
                    urdf_data, 
                    end_link_name="fr_link6",
                    root_link_name="fr_base_link",
                ))

# set scene and load robot
scene = sapien.Scene()
scene.add_ground(0)
scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=True)

viewer = scene.create_viewer()
viewer.set_camera_xyz(x=0.4, y=0.22, z=1.5)
viewer.set_camera_rpy(r=0, p=-0.8, y=2.45)

# load robot
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(urdf_path)

robot.set_root_pose(
            sapien.Pose(
                [0, -0.65, 0],
                [1, 0, 0, 1]
            )
        )

active_joints = robot.get_active_joints()
for joint in active_joints:
    joint.set_drive_property(
        stiffness=1000,
        damping=200)


left_planner = mplib.Planner(
    urdf = urdf_path,
    srdf = srdf_path,
    move_group= "fl_link6",
)
right_planner = mplib.Planner(
    urdf = urdf_path,
    srdf = srdf_path,
    move_group= "fr_link6",
)
robot_pose_in_world = [0,-0.65,0,1,0,0,1] 
left_planner.set_base_pose(robot_pose_in_world)
right_planner.set_base_pose(robot_pose_in_world)

# 获取机器人的所有关节
all_joints = robot.get_joints()
left_pose_test = robot.find_joint_by_name('fl_joint6').global_pose
right_pose_test = robot.find_joint_by_name('fr_joint6').global_pose

left_endpose = endpose_transform(joint=all_joints[42], gripper_val=0.045)
right_endpose = endpose_transform(joint=all_joints[43], gripper_val=0.045)

global_pose = np.array([left_endpose["x"],left_endpose["y"],left_endpose["z"],left_endpose["roll"],
                                                left_endpose["pitch"],left_endpose["yaw"],left_endpose["gripper"],
                                                right_endpose["x"],right_endpose["y"],right_endpose["z"],right_endpose["roll"],
                                                right_endpose["pitch"],right_endpose["yaw"],right_endpose["gripper"],])

left_arm_joint_id = [6,14,18,22,26,30]
right_arm_joint_id = [7,15,19,23,27,31]

a = robot.get_qpos()
qpos_left = np.array([a[6], a[14], a[18], a[22], a[26], a[30]], dtype=np.float64)  # 先转换数据类型为float64

all_pose_left = pk_chain_left.forward_kinematics(qpos_left, end_only=False)
pose_mats_l = all_pose_left["fl_link6"].get_matrix()

base_pose_matrix_tensor = torch.tensor(base_pose_matrix, dtype=torch.float32, device=device)  # 确保转换时指定设备
pose_mats_l = pose_mats_l.to(device)  # 如果还不在 GPU 上，转移它

pose_mats_left = base_pose_matrix_tensor @ pose_mats_l  # 执行 Tensor 的矩阵乘法

#xyz_l  = pose_mats_l[:,:3,3]
xyz_l = pose_mats_left[:,:3,3].cpu().numpy()


qpos_right = np.array([a[i] for i in right_arm_joint_id])
all_pose_right = pk_chain_right.forward_kinematics(qpos_right, end_only=False)
pose_mats_r = all_pose_right["fr_link6"].get_matrix().to(device)
pose_mats_right = base_pose_matrix_tensor @ pose_mats_r
xyz_r = pose_mats_right[:,:3,3].cpu().numpy()


while not viewer.closed:
    for _ in range(4):
        scene.step()
    scene.update_render()
    viewer.render()


# 研究坐标转换
T_base_to_fl = torch.tensor([[1, 0, 0, 0.233],
                                        [0, 1, 0, 0.300],
                                        [0, 0, 1, 0.6275],
                                        [0, 0, 0, 1]], dtype=dtype, device=self.device)






