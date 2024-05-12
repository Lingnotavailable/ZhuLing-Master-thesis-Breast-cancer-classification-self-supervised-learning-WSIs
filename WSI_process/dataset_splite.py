import os
import shutil
from sklearn.model_selection import train_test_split

# 定义路径
source_dir = 'data\IMG-SPLIT\All'  # 源目录
target_dir = 'data\IMG-SPLIT'  # 目标目录
classes = os.listdir(source_dir)  # 类别名

# 定义分割比例
train_size = 0.7
val_size = 0.15
test_size = 0.15

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

for cls in classes:
    os.makedirs(os.path.join(target_dir, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val', cls), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test', cls), exist_ok=True)

    # 获取当前类别下所有图片
    images = os.listdir(os.path.join(source_dir, cls))

    # 分割数据集
    train_val, test = train_test_split(images, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size / (train_size + val_size), random_state=42)

    # 复制文件
    for image in train:
        shutil.copy(os.path.join(source_dir, cls, image), os.path.join(target_dir, 'train', cls, image))
    for image in val:
        shutil.copy(os.path.join(source_dir, cls, image), os.path.join(target_dir, 'val', cls, image))
    for image in test:
        shutil.copy(os.path.join(source_dir, cls, image), os.path.join(target_dir, 'test', cls, image))

print("数据集分割完成。")
