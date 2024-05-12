import os
import shutil
from random import sample


def count_images(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def copy_images(source_directory, target_directory, num_images):
    files = [name for name in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, name))]
    selected_files = sample(files, num_images)
    for file in selected_files:
        shutil.copy(os.path.join(source_directory, file), target_directory)

def copy_images_all(source_directory, target_directory):
    files = [name for name in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, name))]
    for file in files:
        shutil.copy(os.path.join(source_directory, file), target_directory)


path = 'C:\Users\UEANU\Desktop\JULY\DL\mmpretrain\data\Dataset-Finetuning-new\Dataset100'
list = ['train', 'test', 'val']
name = ['Invasive', 'Benign', 'InSitu']
for item in list:
    path = os.path.join(path, item)

