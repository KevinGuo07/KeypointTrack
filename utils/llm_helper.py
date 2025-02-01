# keguide/utils/llm_helper.py

import numpy as np


class LLMDecisionMaker: 
    def __init__(self):
        """
        初始化决策者。
        """
        # 用于记录对某些方块是否已经彻底忽略（不再计算距离）
        self.ignore_box_distance = set()  

        # 用于记录上一次左右手分别的状态（仅针对被关心的方块ID）
        # left_state 取值:
        #   1: distance_left < 0.22 and left_gripper_open
        #   2: distance_left < 0.22 and not left_gripper_open
        #   0: 其他
        # right_state 取值:
        #   3: distance_right < 0.22 and right_gripper_open
        #   4: distance_right < 0.22 and not right_gripper_open
        #   0: 其他
        self.prev_left_state = {}
        self.prev_right_state = {}

    def get_decision(self, current_state, box_positions, short_history=None, prompt=None, **kwargs):
        # 初始化 llm_output
        llm_output = {
            "enable_affordance_guidance": {
                "enable": False,
                "enable_left_arm": {
                    "enable": False,
                    "point": [0.0, 0.0, 0.0],
                },
                "enable_right_arm": {
                    "enable": False,
                    "point": [0.0, 0.0, 0.0],
                }
            },
        }

        # 获取机械臂末端执行器的位置和夹爪状态
        ee_pose = current_state.get('end_pose', [])
        if len(ee_pose) < 14:
            print("Error: end_pose has insufficient length")
            return llm_output

        # 提取左右机械臂末端的位置和夹爪状态
        left_ee_pose = np.array(ee_pose[0:3])     # [x, y, z]
        left_gripper_val = ee_pose[6]            # 左夹爪状态值
        right_ee_pose = np.array(ee_pose[7:10])  # [x, y, z]
        right_gripper_val = ee_pose[13]          # 右夹爪状态值

        # 判断夹爪是否打开的阈值 xx
        OPEN_THRESHOLD = 0.04 
        left_gripper_open = left_gripper_val > OPEN_THRESHOLD
        right_gripper_open = right_gripper_val > OPEN_THRESHOLD

        # 只处理 actor_id == 66
        target_ids = [66, 67]
        # target_ids = [66]

        for target_box_id in target_ids:
            if target_box_id not in box_positions:
                continue

            # 如果已经在忽略列表中，则直接跳过该目标
            if target_box_id in self.ignore_box_distance:
                continue

            # 初始化上一帧左右手状态（如果尚无记录）
            if target_box_id not in self.prev_left_state:
                self.prev_left_state[target_box_id] = 0
            if target_box_id not in self.prev_right_state:
                self.prev_right_state[target_box_id] = 0

            box_pos = box_positions[target_box_id]
            box_position = np.array(box_pos)

            # 计算与左右臂末端的距离
            distance_left = np.linalg.norm(left_ee_pose - box_position)
            distance_right = np.linalg.norm(right_ee_pose - box_position)

            # ======== 根据距离和夹爪状态判断当前左/右手的状态 ========
            # 左手状态: 1, 2, or 0
            if distance_left < 0.3 and left_gripper_open:
                left_state = 1
            elif distance_left < 0.3 and not left_gripper_open:
                left_state = 2
            else:
                left_state = 0

            # 右手状态: 3, 4, or 0
            if distance_right < 0.3 and right_gripper_open:
                right_state = 3
            elif distance_right < 0.3 and not right_gripper_open:
                right_state = 4
            else:
                right_state = 0

            # ======== 检查是否发生了“从状态1->2”或“从状态3->4”的转换 ========
            prev_l = self.prev_left_state[target_box_id]
            prev_r = self.prev_right_state[target_box_id]

            # 从1->2
            if prev_l == 1 and left_state == 2:
                print(f"[Box {target_box_id}] Detected left arm state transition: 1 -> 2, ignoring box")
                llm_output["enable_affordance_guidance"]["enable"] = False
                self.ignore_box_distance.add(target_box_id)
                # 继续处理下一个 box_id
                continue

            # 从3->4
            if prev_r == 3 and right_state == 4:
                print(f"[Box {target_box_id}] Detected right arm state transition: 3 -> 4, ignoring box")
                llm_output["enable_affordance_guidance"]["enable"] = False
                self.ignore_box_distance.add(target_box_id)
                # 继续处理下一个 box_id
                continue

            # ======== 如果没有发生需要“永久忽略”的转换，则执行新逻辑 ========
            # 需求：在“某个夹爪开启对某目标的引导”前，比较“另一只夹爪到该目标的距离”是否 > “当前夹爪到该目标的距离”
            # 只有在另一只夹爪距离更大时才执行引导，否则忽略。
            # -----------------------------------------------------------
            # 左手想要引导
            if distance_left < 0.3 and left_gripper_open:
                # 先检查另一只手(右手)是否离得更远
                if distance_right > distance_left:
                    print(f"[Box {target_box_id}] Launching guidance for left arm")
                    llm_output["enable_affordance_guidance"]["enable"] = True
                    llm_output["enable_affordance_guidance"]["enable_left_arm"]["enable"] = True
                    llm_output["enable_affordance_guidance"]["enable_left_arm"]["point"] = box_pos
                else:
                    print(f"[Box {target_box_id}] Left arm wants guidance but right arm is closer or equal, skip")

            # 左手靠近但夹爪没开
            elif distance_left < 0.3 and not left_gripper_open:
                print(f"[Box {target_box_id}] Left gripper is not open, skip guidance for left arm")
                # 可以根据需要决定是否要设置 enable=False (这里与原逻辑保持一致)
                llm_output["enable_affordance_guidance"]["enable"] = False

            # 右手想要引导
            if distance_right < 0.3 and right_gripper_open:
                # 先检查另一只手(左手)是否离得更远
                if distance_left > distance_right:
                    print(f"[Box {target_box_id}] Launching guidance for right arm")
                    llm_output["enable_affordance_guidance"]["enable"] = True
                    llm_output["enable_affordance_guidance"]["enable_right_arm"]["enable"] = True
                    llm_output["enable_affordance_guidance"]["enable_right_arm"]["point"] = box_pos
                else:
                    print(f"[Box {target_box_id}] Right arm wants guidance but left arm is closer or equal, skip")

            # 右手靠近但夹爪没开
            elif distance_right < 0.3 and not right_gripper_open:
                print(f"[Box {target_box_id}] Right gripper is not open, skip guidance for right arm")
                # 可以根据需要决定是否要设置 enable=False (这里与原逻辑保持一致)
                llm_output["enable_affordance_guidance"]["enable"] = False

            # 如果左右手都不符合接近条件，则禁用引导
            if distance_left >= 0.3 and distance_right >= 0.3:
                llm_output["enable_affordance_guidance"]["enable"] = False
                llm_output["enable_affordance_guidance"]["enable_left_arm"]["enable"] = False
                llm_output["enable_affordance_guidance"]["enable_right_arm"]["enable"] = False

            # 最后记录本帧的左右手状态
            self.prev_left_state[target_box_id] = left_state
            self.prev_right_state[target_box_id] = right_state

        return llm_output

    def get_decision_pre(self, current_state, box_positions, short_history=None, prompt=None, **kwargs):
        
        # 初始化 llm_output 模板
        llm_output = {
            "enable_affordance_guidance": {
                "enable": False,
                "enable_left_arm": {
                    "enable": False,
                    "point": [0.0, 0.0, 0.0],
                },
                "enable_right_arm": {
                    "enable": False,
                    "point": [0.0, 0.0, 0.0],
                }
            },
        }

        # 获取机械臂末端执行器的位置和夹爪状态
        ee_pose = current_state.get('end_pose', [])
        if len(ee_pose) < 14:
            print("Error: end_pose has insufficient length")
            return llm_output

        # 提取左右机械臂末端的位置和夹爪状态
        left_ee_pose = np.array(ee_pose[0:3])       # [x, y, z]
        left_gripper_val = ee_pose[6]              # 左夹爪状态值
        right_ee_pose = np.array(ee_pose[7:10])    # [x, y, z]
        right_gripper_val = ee_pose[13]            # 右夹爪状态值

        # 定义夹爪是否打开的阈值
        OPEN_THRESHOLD = 0.036  # 根据实际情况调整

        # 判断夹爪是否打开
        left_gripper_open = left_gripper_val > OPEN_THRESHOLD
        right_gripper_open = right_gripper_val > OPEN_THRESHOLD

        # 遍历所有方块，查找是否有未被引导且距离夹爪小于0.22的方块
        for actor_id, box_pos in box_positions.items():
            if actor_id <= 65:
                continue  # 只考虑ID>65的物体
            if actor_id in self.guided_boxes:
                continue  # 已经被引导过的方块跳过

            box_position = np.array(box_pos)

            # 计算与左臂末端的距离
            distance_left = np.linalg.norm(left_ee_pose - box_position)
            print(f"Distance to box {actor_id} from left arm: {distance_left}")
            # 计算与右臂末端的距离
            distance_right = np.linalg.norm(right_ee_pose - box_position)
            print(f"Distance to box {actor_id} from right arm: {distance_right}")

            # 判断左臂是否需要引导
            if distance_left < 0.22 and left_gripper_open:
                print(f"Launching guidance for left arm to box {actor_id}")
                # 启用左臂引导
                llm_output["enable_affordance_guidance"]["enable"] = True
                llm_output["enable_affordance_guidance"]["enable_left_arm"]["enable"] = True
                llm_output["enable_affordance_guidance"]["enable_left_arm"]["point"] = box_pos
                # 记录该方块已被引导
                # self.guided_boxes.add(actor_id)
                # 由于一个方块只能被一只臂引导，跳出循环
                break
            else:
                # print(f"Left arm not guiding to box {actor_id}")
                llm_output["enable_affordance_guidance"]["enable"] = False
                llm_output["enable_affordance_guidance"]["enable_left_arm"]["enable"] = False

            # 判断右臂是否需要引导
            if distance_right < 0.22 and right_gripper_open:
                # 启用右臂引导
                print(f"Launching guidance for right arm to box {actor_id}")
                llm_output["enable_affordance_guidance"]["enable"] = True
                llm_output["enable_affordance_guidance"]["enable_right_arm"]["enable"] = True
                llm_output["enable_affordance_guidance"]["enable_right_arm"]["point"] = box_pos
                # 记录该方块已被引导
                # self.guided_boxes.add(actor_id)
                # 由于一个方块只能被一只臂引导，跳出循环
                break
            else:
                # print(f"Right arm not guiding to box {actor_id}")
                llm_output["enable_affordance_guidance"]["enable"] = False
                llm_output["enable_affordance_guidance"]["enable_right_arm"]["enable"] = False

        return llm_output
