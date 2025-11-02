import os
import shutil
from pathlib import Path

# -------------------------- 1. 配置路径（必须修改为你的实际路径） --------------------------
HEALTHY_IMG_DIR = r"E:\YOLOGrape\Grape_Dataset\raw_data\Healthy"  # 健康叶片图片原始文件夹
HEALTHY_XML_DIR = r"E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\healthy_voc"  # 健康类XML标注文件夹（从makesense解压的路径）
JPEG_DIR = r"E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\VOC2007\JPEGImages"  # VOC的JPEGImages路径
ANNO_DIR = r"E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\VOC2007\Annotations"  # VOC的Annotations路径
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]  # 图片格式

# -------------------------- 2. 复制健康图片到JPEGImages --------------------------
img_copied = 0
for img_file in os.listdir(HEALTHY_IMG_DIR):
    img_ext = Path(img_file).suffix.lower()
    if img_ext not in IMAGE_EXTENSIONS:
        continue  # 跳过非图片文件

    src_img_path = os.path.join(HEALTHY_IMG_DIR, img_file)
    dst_img_path = os.path.join(JPEG_DIR, img_file)

    # 跳过已存在的文件（避免覆盖）
    if os.path.exists(dst_img_path):
        print(f"⚠️ 图片 {img_file} 已存在，跳过")
        continue

    # 复制图片
    shutil.copy2(src_img_path, dst_img_path)  # copy2保留文件元信息
    img_copied += 1

print(f"\n✅ 成功复制 {img_copied} 张健康图片到 JPEGImages 文件夹")

# -------------------------- 3. 复制健康类XML标注到Annotations --------------------------
xml_copied = 0
for xml_file in os.listdir(HEALTHY_XML_DIR):
    if not xml_file.endswith(".xml"):
        continue  # 跳过非XML文件

    src_xml_path = os.path.join(HEALTHY_XML_DIR, xml_file)
    dst_xml_path = os.path.join(ANNO_DIR, xml_file)

    # 跳过已存在的文件
    if os.path.exists(dst_xml_path):
        print(f"⚠️ 标注 {xml_file} 已存在，跳过")
        continue

    # 复制标注
    shutil.copy2(src_xml_path, dst_xml_path)
    xml_copied += 1

print(f"✅ 成功复制 {xml_copied} 个健康类XML标注到 Annotations 文件夹")

# -------------------------- 4. 验证图片与标注是否匹配 --------------------------
# 获取JPEGImages中健康图片的文件名（去后缀）
jpeg_healthy_names = [Path(f).stem for f in os.listdir(JPEG_DIR) if Path(f).stem.startswith("healthy")]
# 获取Annotations中健康标注的文件名（去后缀）
anno_healthy_names = [Path(f).stem for f in os.listdir(ANNO_DIR) if Path(f).stem.startswith("healthy")]

# 检查缺失的标注/图片
missing_anno = [name for name in jpeg_healthy_names if name not in anno_healthy_names]
missing_img = [name for name in anno_healthy_names if name not in jpeg_healthy_names]

if missing_anno:
    print(f"\n❌ 警告：以下健康图片缺少标注：{missing_anno}")
else:
    print("\n✅ 所有健康图片均有对应标注，无缺失")

if missing_img:
    print(f"❌ 警告：以下健康标注缺少图片：{missing_img}")
else:
    print("✅ 所有健康标注均有对应图片，无缺失")

print("\n🎉 健康类图片和标注复制完成！")