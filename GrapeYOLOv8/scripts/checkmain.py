import xml.etree.ElementTree as ET
import os
from pathlib import Path
import shutil
import re

# 类别名称
CLASS_NAMES = ['black_rot', 'blight', 'black_measles', 'Healthy']


def convert_box(size, box):
    """将VOC格式边界框转换为YOLO格式"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def clean_image_id(image_id):
    """清理图像ID，移除特殊字符和空格"""
    # 移除括号和空格等特殊字符
    cleaned_id = re.sub(r'[^\w\s-]', '', image_id).strip()
    # 将多个空格替换为单个下划线
    cleaned_id = re.sub(r'\s+', '_', cleaned_id)
    return cleaned_id


def find_image_file(voc_dir, image_id):
    """查找图像文件，尝试多种可能的文件名格式"""
    # 先尝试原始ID
    for ext in ['.jpg', '.png', '.jpeg']:
        path = os.path.join(voc_dir, 'JPEGImages', f'{image_id}{ext}')
        if os.path.exists(path):
            return path

    # 如果原始ID找不到，尝试清理后的ID
    cleaned_id = clean_image_id(image_id)
    for ext in ['.jpg', '.png', '.jpeg']:
        path = os.path.join(voc_dir, 'JPEGImages', f'{cleaned_id}{ext}')
        if os.path.exists(path):
            return path

    return None


def find_xml_file(voc_dir, image_id):
    """查找XML文件，尝试多种可能的文件名格式"""
    # 先尝试原始ID
    xml_path = os.path.join(voc_dir, 'Annotations', f'{image_id}.xml')
    if os.path.exists(xml_path):
        return xml_path

    # 如果原始ID找不到，尝试清理后的ID
    cleaned_id = clean_image_id(image_id)
    xml_path = os.path.join(voc_dir, 'Annotations', f'{cleaned_id}.xml')
    if os.path.exists(xml_path):
        return xml_path

    return None


def convert_annotation(voc_dir, image_id, output_dir):
    """转换单个XML标注文件"""
    xml_file = find_xml_file(voc_dir, image_id)

    if not xml_file:
        print(f"警告：找不到XML文件 {image_id}.xml")
        return False

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # 使用清理后的ID作为输出文件名
        cleaned_id = clean_image_id(image_id)
        out_file_path = os.path.join(output_dir, f'{cleaned_id}.txt')

        with open(out_file_path, 'w', encoding='utf-8') as out_file:
            for obj in root.iter('object'):
                # 处理可能不存在的difficult标签
                difficult_elem = obj.find('difficult')
                difficult = difficult_elem.text if difficult_elem is not None else '0'

                cls = obj.find('name').text
                if cls not in CLASS_NAMES or int(difficult) == 1:
                    continue
                cls_id = CLASS_NAMES.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text),
                     float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                bb = convert_box((w, h), b)
                out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

        return True

    except Exception as e:
        print(f"处理文件 {xml_file} 时出错: {e}")
        return False


def process_dataset(voc_dir, output_base_dir):
    """处理整个数据集"""
    sets = ['train', 'val', 'test']

    for image_set in sets:
        # 读取图片列表
        set_file = os.path.join(voc_dir, 'ImageSets', 'Main', f'{image_set}.txt')
        if not os.path.exists(set_file):
            print(f"警告：{set_file} 不存在，跳过")
            continue

        with open(set_file, 'r') as f:
            image_ids = f.read().strip().split()

        # 创建输出目录
        label_dir = os.path.join(output_base_dir, 'labels', image_set)
        image_dir = os.path.join(output_base_dir, 'images', image_set)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # 转换标注并复制图片
        converted_count = 0
        for image_id in image_ids:
            # 转换标注
            if convert_annotation(voc_dir, image_id, label_dir):
                # 复制图片
                src_img = find_image_file(voc_dir, image_id)
                if src_img:
                    cleaned_id = clean_image_id(image_id)
                    ext = os.path.splitext(src_img)[1]
                    dst_img = os.path.join(image_dir, f'{cleaned_id}{ext}')
                    shutil.copy(src_img, dst_img)
                    converted_count += 1
                else:
                    print(f"警告：找不到图片文件 {image_id}")
            else:
                print(f"警告：无法转换标注 {image_id}")

        print(f"{image_set} 集处理完成：{converted_count}/{len(image_ids)} 张图片")


# 执行转换 - 修正路径
VOC_DIR = "E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\VOC2007"  # 修正为VOC2007
OUTPUT_DIR = "E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\GrapeYOLOv8\data"

# 检查VOC目录是否存在
if not os.path.exists(VOC_DIR):
    print(f"错误：VOC目录不存在: {VOC_DIR}")
    print("请检查路径是否正确")
else:
    process_dataset(VOC_DIR, OUTPUT_DIR)
    print("数据集转换完成！")