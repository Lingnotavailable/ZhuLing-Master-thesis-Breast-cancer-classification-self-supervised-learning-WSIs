import os
import random
import shutil

def copy_random_images(src_folder, dst_folder, n):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = [f for f in os.listdir(src_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random_files = random.sample(files, n)

    for file in random_files:
        src_file = os.path.join(src_folder, file)
        dst_file = os.path.join(dst_folder, file)
        shutil.copy2(src_file, dst_file)

src_folder = 'new_dataset\Malignant_croped'
dst_folder = 'breaKHis_dataset\Malignant'
n = 10000  
copy_random_images(src_folder, dst_folder, n)

src_folder = 'new_dataset\Benign_croped'
dst_folder = 'breaKHis_dataset\Benign'
n = 10000  
copy_random_images(src_folder, dst_folder, n)


