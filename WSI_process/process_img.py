import os.path

from PIL import Image

image_folder = 'data\ICIAR2018_BACH_Challenge\Photos'
output_dir = 'data\IMG-SPLIT\All'

class_names = ['Benign','InSitu','Invasive','Normal']


def crop_image(image_folder_path,image_name,image_folder):
    # 打开原始图片
    original_image = Image.open(os.path.join(image_folder_path,image_name))  # 替换为您的图片路径

    # 设置子图像的尺寸
    tile_width, tile_height = 256, 256
    x_tiles = original_image.width // tile_width
    y_tiles = original_image.height // tile_height

    # 循环遍历每个子图像的坐标
    for x in range(x_tiles):
        for y in range(y_tiles):
            # 计算当前子图像的坐标
            left = x * tile_width
            upper = y * tile_height
            right = left + tile_width
            lower = upper + tile_height
            tile = original_image.crop((left, upper, right, lower))
            tile_path = os.path.join(image_folder,f'{image_name[:-4]}_{x}_{y}.png')  # 保存为PNG格式
            tile.save(tile_path)

for class_name in class_names:
    image_sub_folder = os.path.join(image_folder,class_name)
    image_list = os.listdir(image_sub_folder)
    for index,image_name in enumerate(image_list):
        crop_image(image_sub_folder,image_name,os.path.join(output_dir,class_name))
        print(f'{index}/{class_name}')