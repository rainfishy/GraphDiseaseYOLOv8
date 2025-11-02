import os

base_dir = "GrapeYOLOv8"
folders = [
    "data/images/train", "data/images/val", "data/images/test",
    "data/labels/train", "data/labels/val", "data/labels/test",
    "models", "runs", "scripts"
]

for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)
print("文件夹结构创建完成！")