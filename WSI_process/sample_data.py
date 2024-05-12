import os
import shutil
import random

# 源文件夹和目标文件夹的路径
source_folder = r'data\Dataset-Finetuning-new\Dataset100\train'
destination_folder = r'data\Dataset-Finetuning-new\Dataset10\train'

def move_file(source_folder,destination_folder,percentage):

    # 获取源文件夹中的所有文件列表
    files = [file for file in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, file))]

    # 计算要移动的文件数量，约为总文件数的5%
    num_files_to_move = max(1, int(len(files)*percentage))

    # 随机选择文件
    files_to_move = random.sample(files, num_files_to_move)

    # 将选中的文件移动到目标文件夹
    for file in files_to_move:
        shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder, file))

    print(f"Moved {len(files_to_move)} files from {source_folder} to {destination_folder}.")

class_names = ['Benign','InSitu','Invasive','Normal']

for class_name in class_names:
    sf = os.path.join(source_folder,class_name)
    tf = os.path.join(destination_folder,class_name)
    move_file(sf,tf,0.1)


    
