import os
import random

# 指定目标文件夹路径

def remove(pre,fp,num_to_keep):
    folder_path = fp
    # 指定前缀
    prefix = pre
    # 列出所有以特定前缀开头的图片
    files = [f for f in os.listdir(folder_path) if f.startswith(prefix)]

    # 根据比例计算要删除的图片数量
    num_to_delete = len(files)-num_to_keep

    # 随机选择要删除的图片
    files_to_delete = random.sample(files, num_to_delete)

    # 删除选中的图片
    for file in files_to_delete:
        os.remove(os.path.join(folder_path, file))
        print(f"Deleted {file}")

def remove_(pre,fp,num_to_keep):
    # 获取所有不以'brea'开头的图片
    images = [img for img in os.listdir(fp) if not img.startswith(pre) and img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    # 设置要删除的图片数量
    number_to_delete = len(images)-num_to_keep  # 可以根据需要修改这个数量
    # 确保删除的数量不超过列表中的图片数量
    number_to_delete = min(number_to_delete, len(images))
    # 随机选择要删除的图片
    selected_images = random.sample(images, number_to_delete)
    # 删除选中的图片
    for img in selected_images:
        os.remove(os.path.join(fp, img))
        print(f'图片 {img} 已被删除')


data_path  = 'data\Dataset-Finetuning-new'
dataset_list = [f'Dataset{i}' for i in [100]]
sub_path_list = ['train']
for dataset in dataset_list:
        remove('breast',os.path.join(data_path,dataset,'train','Invasive'),1680)
        remove('breast',os.path.join(data_path,dataset,'test','Invasive'),360)
        remove('breast',os.path.join(data_path,dataset,'val','Invasive'),360)
        remove_('breast',os.path.join(data_path,dataset,'train','Invasive'),1680)
        remove_('breast',os.path.join(data_path,dataset,'test','Invasive'),360)
        remove_('breast',os.path.join(data_path,dataset,'val','Invasive'),360)