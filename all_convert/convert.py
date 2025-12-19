import shutil
from pathlib import Path
import h5py
import numpy as np
import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 设置常量
REPO_NAME = "rokae_lora"
DATA_DIR = "/media/dainanjun/T9/new"  # 修改为包含所有小文件夹的大文件夹路径
OUTPUT_DIR = "/home/dainanjun/robot/openpi/all_convert"  # 这个是输出路径，仍然保留

def main():
    # 设置路径
    data_dir = Path(DATA_DIR)  # 大文件夹路径
    output_dir = Path(OUTPUT_DIR)  # 输出目录

    # 获取所有子文件夹（每个任务文件夹）
    task_folders = [folder for folder in data_dir.iterdir() if folder.is_dir()]
    
    # 输出路径：output/rokae_xmatepro7
    dataset_root = output_dir / REPO_NAME
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    target_h, target_w = 256, 256

    # 创建 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=str(dataset_root),  # 直接传本地路径
        robot_type="rokae",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (target_h, target_w, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (target_h, target_w, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 遍历每个小文件夹（任务）
    for task_folder in task_folders:
        # 获取该任务文件夹中的所有 .h5 文件
        h5_files = sorted(task_folder.glob("*.h5"))
        
        if len(h5_files) == 0:
            print(f"Warning: No .h5 files found in task folder: {task_folder.name}")
            continue  # 如果该文件夹内没有 .h5 文件，则跳过

        # 设置 task 名称为当前小文件夹的名称
        fixed_task = task_folder.name

        print(f"Processing task folder: {fixed_task}")

        # 处理每一个 h5 文件
        for h5_path in h5_files:
            with h5py.File(str(h5_path), "r") as f:
                exterior = f["observation/exterior_image_1_left"][...]
                wrist = f["observation/wrist_image"][...]
                joint = f["observation/joint_position"][...].astype(np.float32)
                gripper = f["observation/gripper_position"][...].astype(np.float32)

                act_joint = f["action/joint_position"][...].astype(np.float32)
                act_gripper = f["action/gripper_position"][...].astype(np.float32)

            T = exterior.shape[0]

            for t in range(T):
                ext_img = cv2.resize(exterior[t], (target_w, target_h))
                wrist_img = cv2.resize(wrist[t], (target_w, target_h))

                actions = np.concatenate([act_joint[t], act_gripper[t]], axis=-1)

                # 添加帧数据到数据集
                dataset.add_frame(
                    {
                        "exterior_image_1_left": ext_img,
                        "wrist_image_left": wrist_img,
                        "joint_position": joint[t],
                        "gripper_position": gripper[t],
                        "actions": actions,
                        "task": fixed_task,  # 使用当前任务文件夹的名称作为 task
                    }
                )

            dataset.save_episode()

            # 输出每个文件的成功处理信息
            print(f"Successfully processed: {h5_path.name} in task: {fixed_task}")

    print(f"✅ LeRobot dataset written to: {dataset_root.resolve()}")

# 直接调用 main 函数，无需命令行参数
if __name__ == "__main__":
    main()
