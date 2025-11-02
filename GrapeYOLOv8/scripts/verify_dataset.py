import os
from pathlib import Path

# 配置VOC数据集路径
VOC_DIR = r"E:\VOC2007"
JPEG_DIR = os.path.join(VOC_DIR, "Annotations")
ANNO_DIR = os.path.join(VOC_DIR, "JPEGImages")

# 检查图片与标注的一致性
missed_anno = []
missed_img = []

# 收集所有图片文件名（无后缀）
img_names = [Path(f).stem for f in os.listdir(JPEG_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]
# 收集所有标注文件名（无后缀）
anno_names = [Path(f).stem for f in os.listdir(ANNO_DIR) if f.endswith(".xml")]

# 检查缺少标注的图片
for img in img_names:
    if img not in anno_names:
        missed_anno.append(img)
# 检查缺少图片的标注
for anno in anno_names:
    if anno not in img_names:
        missed_img.append(anno)

# 输出结果
if missed_anno:
    print(f"⚠️ 缺少标注的图片：{missed_anno}")
else:
    print("✅ 所有图片都有对应标注")

if missed_img:
    print(f"⚠️ 缺少图片的标注：{missed_img}")
else:
    print("✅ 所有标注都有对应图片")