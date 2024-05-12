

import h5py
from PIL import Image
import numpy as np

# 替换为你的h5文件路径


# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'C:\Users\UEANU\Desktop\JULY\WSI\openslide-win64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


from openslide import OpenSlide

svs_path = 'data/WSI-BACH'
h5_path = 'data/WSI-SPLIT/patches'
svs_list = os.listdir(svs_path)
for svs_file_name in svs_list:
    svs_id = svs_file_name[:4]
    slide = OpenSlide(os.path.join(svs_path,svs_file_name))
    h5_file_path = os.path.join(h5_path,svs_id+'.h5')
    with h5py.File(h5_file_path, 'r') as h5_file:
        data = h5_file['coords']
        cordinate_list = np.array(data[:])
    for index, coordinates in enumerate(cordinate_list):
        region = slide.read_region((coordinates[0], coordinates[1]), 1, (230, 230))
        region_image = Image.fromarray(np.array(region)[:, :, :3])  # 移除Alpha通道
        region_image = region_image.resize((256,256))
        save_path = os.path.join('data/WSI-SPLIT/splited_images',f'{svs_id}_{index}.png')
        region_image.save(save_path)
        print(index)
print("所有图像已保存至 'save' 文件夹。")