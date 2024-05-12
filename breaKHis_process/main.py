import os

from PIL import Image

path = 'BreaKHis_v1/histology_slides/breast/benign/SOB'
target_path = 'new_dataset/Benign'
folder_list = os.listdir(path)
number = 1
for folder in folder_list:
    sub_folder_list = os.listdir(os.path.join(path, folder))
    for sub_folder in sub_folder_list:
        image_list = os.listdir(os.path.join(path, folder, sub_folder,'40X'))
        for image in image_list:
            img = Image.open(os.path.join(path, folder, sub_folder,'40X', image))
            img.save(os.path.join(target_path,f'{number}.png'))
            number += 1
            print(number)
