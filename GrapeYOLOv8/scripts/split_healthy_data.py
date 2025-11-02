import os
import random
from pathlib import Path

# -------------------------- 1. 配置路径（必须修改为你的实际路径） --------------------------
HEALTHY_IMG_DIR = r"E:\YOLOGrape\Grape_Dataset\raw_data\Healthy"  # 健康叶片图片原始文件夹
MAIN_DIR = r"E:\YOLOGrape\Grape_Dataset\VOC2007\VOC2007\ImageSets\Main"  # VOC的ImageSets/Main路径
IMAGE_EXTENSIONS = [".jpg"]  # 图片格式（根据你的文件调整）

# -------------------------- 2. 读取健康图片文件名（去后缀） --------------------------
# 获取所有健康图片的文件名（不含后缀）
healthy_img_names = []
for img_file in os.listdir(HEALTHY_IMG_DIR):
    img_ext = Path(img_file).suffix.lower()
    if img_ext in IMAGE_EXTENSIONS:
        img_name_no_ext = Path(img_file).stem  # 去掉后缀，如"healthy_001.jpg"→"healthy_001"
        healthy_img_names.append(img_name_no_ext)

# 打乱顺序（保证划分随机性）
random.seed(42)  # 固定随机种子，确保每次划分结果一致
random.shuffle(healthy_img_names)
total = len(healthy_img_names)
print(f"健康叶片总数：{total} 张")

# -------------------------- 3. 按8:1:1划分子集 --------------------------
train_num = int(total * 0.8)
val_num = int(total * 0.1)
test_num = total - train_num - val_num

healthy_train = healthy_img_names[:train_num]
healthy_val = healthy_img_names[train_num:train_num+val_num]
healthy_test = healthy_img_names[train_num+val_num:]

print(f"健康类训练集：{len(healthy_train)} 张，验证集：{len(healthy_val)} 张，测试集：{len(healthy_test)} 张")

# -------------------------- 4. 生成健康类专属划分文件（healthy_train.txt等） --------------------------
# 定义生成文件的函数
def write_to_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for name in data:
            f.write(f"{name}\n")  # 每行一个文件名（无后缀）

# 生成健康类划分文件
write_to_file(healthy_train, os.path.join(MAIN_DIR, "healthy_train.txt"))
write_to_file(healthy_val, os.path.join(MAIN_DIR, "healthy_val.txt"))
write_to_file(healthy_test, os.path.join(MAIN_DIR, "healthy_test.txt"))
print("✅ 健康类专属划分文件已生成")

# -------------------------- 5. 更新总划分文件（合并病害类与健康类） --------------------------
# 读取原有病害类划分文件（若文件为空，直接用健康类数据）
def read_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]  # 去空行
    return []

# 读取原有病害类的train/val/test
disease_train = read_file(os.path.join(MAIN_DIR, "train.txt"))
disease_val = read_file(os.path.join(MAIN_DIR, "val.txt"))
disease_test = read_file(os.path.join(MAIN_DIR, "test.txt"))

# 合并（病害类 + 健康类）并去重（避免重复添加）
new_train = list(set(disease_train + healthy_train))
new_val = list(set(disease_val + healthy_val))
new_test = list(set(disease_test + healthy_test))

# 重新写入总划分文件
write_to_file(new_train, os.path.join(MAIN_DIR, "train.txt"))
write_to_file(new_val, os.path.join(MAIN_DIR, "val.txt"))
write_to_file(new_test, os.path.join(MAIN_DIR, "test.txt"))
print("✅ 总划分文件（train.txt/val.txt/test.txt）已更新，合并病害类与健康类")

print("\n🎉 健康类数据集划分完成！")