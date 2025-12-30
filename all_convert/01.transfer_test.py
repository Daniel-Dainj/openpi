import h5py
import numpy as np
from pathlib import Path

# 设定大文件夹路径，修改为你实际的文件夹路径
folder_path = "/home/dainanjun/rokae_ws/src/vla_control/vla_control/new"  # 大文件夹路径

# 遍历大文件夹中的所有小文件夹（每个任务指令）
for task_folder in Path(folder_path).iterdir():
    if task_folder.is_dir():  # 确保是文件夹
        print(f"Processing task folder: {task_folder.name}")

        # 遍历小文件夹中的所有 .h5 文件
        h5_files = task_folder.glob("*.h5")

        for h5_file in h5_files:
            # 打开 h5 文件
            with h5py.File(h5_file, "r+") as f:
                # 获取 action 和 observation 数据集
                action_joint_position = f["action/joint_position"]
                action_gripper_position = f["action/gripper_position"]
                observation_joint_position = f["observation/joint_position"]
                observation_gripper_position = f["observation/gripper_position"]

                # 确保每个数据集的长度相同
                num_frames = action_joint_position.shape[0]

                # 用下一帧的 observation 替换当前帧的 action
                action_joint_position[: num_frames - 1] = observation_joint_position[1:num_frames]
                action_gripper_position[: num_frames - 1] = observation_gripper_position[1:num_frames]

                # 这里不再删除最后一帧，保留它不变
                # 保存修改后的文件
                print(
                    f"Successfully updated the action/joint_position and action/gripper_position datasets in file: {h5_file.name}"
                )

        # 每个小文件夹处理完后输出成功
        print(f"All files in task folder '{task_folder.name}' have been successfully processed.\n")
