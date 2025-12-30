import os
import pandas as pd
from pathlib import Path


def count_frames_in_parquet_files(folder_path: str):
    # 获取文件夹中的所有子文件夹（每个任务文件夹）
    task_folders = [folder for folder in Path(folder_path).iterdir() if folder.is_dir()]

    print(f"Folder path: {folder_path}")

    # 遍历每个任务文件夹
    for task_folder in task_folders:
        # 获取该任务文件夹中的所有 .parquet 文件
        parquet_files = task_folder.glob("*.parquet")

        # 当前任务文件夹的名字作为 task
        fixed_task = task_folder.name

        print(f"\nProcessing task folder: {fixed_task}")

        total_frames = 0

        # 遍历该任务文件夹中的所有 Parquet 文件
        for file in parquet_files:
            # 读取 Parquet 文件
            df = pd.read_parquet(file)

            # 获取当前文件的帧数（即行数）
            num_frames = len(df)

            # 累加到总帧数
            total_frames += num_frames

            # 输出当前文件的帧数
            print(f"File: {file.name} - Frames: {num_frames}")

        # 输出当前任务文件夹的总帧数
        print(f"Total frames in task '{fixed_task}': {total_frames}")

    print(f"Finished processing all task folders.")


# 替换为你的 Parquet 文件所在的文件夹路径
folder_path = "/home/dainanjun/robot/openpi/all_convert/new_rokae_lora/data"
count_frames_in_parquet_files(folder_path)
