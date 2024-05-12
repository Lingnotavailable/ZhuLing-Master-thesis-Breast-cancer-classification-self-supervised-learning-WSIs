import os
import h5py
from PIL import Image
import numpy as np
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
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_image(svs_file_name, svs_path, h5_path):
    svs_id = svs_file_name[:-4]
    slide = OpenSlide(svs_path+'/'+svs_file_name)
    h5_file_path = h5_path+'/'+svs_id + '.h5'
    with h5py.File(h5_file_path, 'r') as h5_file:
        data = h5_file['coords']
        cordinate_list = np.array(data[:])
    for index, coordinates in enumerate(cordinate_list):
        region = slide.read_region((coordinates[0], coordinates[1]), 0, (230, 230))
        region_image = Image.fromarray(np.array(region)[:, :, :3])  # 移除Alpha通道
        region_image = region_image.resize((256, 256))
        save_path = 'data/WSI-SPLIT/splited_images'+'/'+f'{svs_id}_{index}.png'
        region_image.save(save_path)

svs_path = 'data/WSI-BACH'
h5_path = 'data/WSI-SPLIT/patches'
svs_list = os.listdir(svs_path)

with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(process_image, svs_list, [svs_path]*len(svs_list), [h5_path]*len(svs_list)), total=len(svs_list)))

print("所有图像已保存。")
