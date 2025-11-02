import os
from PIL import Image
import cv2
import numpy as np

# 1. 原始数据集根路径（根据你的本地路径修改，示例为E盘下的raw_data）
raw_data_root = r"E:\YOLOGrape\Grape_Dataset\raw_data"

# 2. 预处理后的数据保存路径（自动创建，避免覆盖原始数据）
processed_data_root = r"E:\YOLOGrape\Grape_Dataset\processed_data"
os.makedirs(processed_data_root, exist_ok=True)  # 确保保存目录存在

# 3. 类别映射（将原始文件夹名称映射为简洁的类别名，便于后续标注和模型识别）
class_mapping = {
    "Black_rot": "Black_rot",
    "Esca_(Black_Measles)": "Black_Measles",
    "Healthy": "Healthy",
    "Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf_blight"
}


def process_image(image_path, save_dir, class_name):
    """处理单张图像：转JPG、统一尺寸、增强质量"""
    try:
        # 1. 打开图像（处理多种格式：PNG、JPG等）
        with Image.open(image_path) as img:
            # 转换为RGB模式（处理PNG透明通道）
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # 2. 统一尺寸为640×640（保持病斑比例，使用LANCZOS插值优化细节）
            img = img.resize((640, 640), Image.Resampling.LANCZOS)

            # 3. 保存为JPG格式（质量95，平衡压缩与细节保留）
            img_name = os.path.basename(image_path)
            img_name = os.path.splitext(img_name)[0] + ".jpg"  # 统一后缀为.jpg
            save_path = os.path.join(save_dir, img_name)
            img.save(save_path, "JPEG", quality=95)

            # 4. 图像增强（高斯模糊去噪 + 锐化突出病斑）
            enhance_image(save_path, save_path)  # 增强后覆盖原保存路径

            print(f"✅ 已处理：{class_name}/{img_name}")
    except Exception as e:
        print(f"❌ 处理失败 {image_path}：{str(e)}")


def enhance_image(input_path, output_path):
    """图像增强：高斯模糊去噪 + 锐化突出病斑"""
    img = cv2.imread(input_path)
    if img is None:
        print(f"⚠️ 无法读取图像 {input_path}")
        return

    # 高斯模糊（核大小5×5，标准差0）去噪
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # 锐化（原图权重2，模糊图权重-1，突出边缘）
    sharpened = cv2.addWeighted(img, 2, blurred, -1, 0)
    # 保存增强后的图像
    cv2.imwrite(output_path, sharpened)


# 遍历每个原始类别文件夹
for raw_class_dir, new_class_name in class_mapping.items():
    raw_class_path = os.path.join(raw_data_root, raw_class_dir)
    if not os.path.exists(raw_class_path):
        print(f"⚠️ 警告：文件夹 {raw_class_path} 不存在，请检查路径！")
        continue

    # 创建预处理后的类别保存目录
    processed_class_dir = os.path.join(processed_data_root, new_class_name)
    os.makedirs(processed_class_dir, exist_ok=True)

    # 获取该类别下所有图像文件
    image_files = [f for f in os.listdir(raw_class_path)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # 批量处理每张图像
    for img_file in image_files:
        img_path = os.path.join(raw_class_path, img_file)
        process_image(img_path, processed_class_dir, new_class_name)

print("\n🎉 数据集预处理完成！所有图像已转换为640×640 JPG格式并增强，保存至：")
print(processed_data_root)