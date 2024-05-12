from PIL import Image
from PIL import Image
import os

# 设置原始图片文件夹路径和输出文件夹路径
original_folder = 'new_dataset\Benign'
output_folder = 'new_dataset\Benign_croped'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

number = 0

# 遍历原始文件夹中的所有文件
for filename in os.listdir(original_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 确保处理的是图片文件
        # 打开图片
        with Image.open(os.path.join(original_folder, filename)) as img:
            # 将图片的长和宽resize成原来的5倍
            img_resized = img.resize((img.width * 5, img.height * 5), Image.ANTIALIAS)
            
            # 计算可以裁剪的256x256区域的数量
            num_crops_x = img_resized.width // 256
            num_crops_y = img_resized.height // 256
            
            # 对每个区域进行裁剪
            for i in range(num_crops_x):
                for j in range(num_crops_y):
                    # 定义裁剪区域的坐标
                    left = i * 256
                    upper = j * 256
                    right = left + 256
                    lower = upper + 256
                    
                    # 裁剪图片
                    cropped_img = img_resized.crop((left, upper, right, lower))
                    
                    # 保存裁剪后的图片
                    base, ext = os.path.splitext(filename)
                    crop_index = j * num_crops_x + i + 1  # 计算裁剪图片的索引
                    output_filename = f"{base}_{crop_index}{ext}"
                    cropped_img.save(os.path.join(output_folder, output_filename))
            
            number+=1
            print(number)
