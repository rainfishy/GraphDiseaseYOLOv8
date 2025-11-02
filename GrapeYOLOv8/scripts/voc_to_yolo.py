import xml.etree.ElementTree as ET
import os
from pathlib import Path
import shutil

# 类别映射（根据你的数据集修改）
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


def convert_annotation(voc_dir, image_id, output_dir):
    """转换单个XML标注文件"""
    # 使用 os.path.join 避免路径问题
    xml_path = os.path.join(voc_dir, 'Annotations', f'{image_id}.xml')
    txt_path = os.path.join(output_dir, f'{image_id}.txt')

    # 检查XML文件是否存在
    if not os.path.exists(xml_path):
        print(f"警告：标注文件不存在 - {xml_path}")
        return False

    try:
        in_file = open(xml_path, encoding='utf-8')
        out_file = open(txt_path, 'w')

        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')

        if size is None:
            print(f"警告：{xml_path} 中没有size标签")
            in_file.close()
            out_file.close()
            return False

        w = int(size.find('width').text)
        h = int(size.find('height').text)

        obj_count = 0
        for obj in root.iter('object'):
            # 检查difficult标签是否存在
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue

            cls = obj.find('name').text
            if cls not in CLASS_NAMES:
                print(f"警告：未知类别 '{cls}' 在文件 {xml_path}")
                continue

            cls_id = CLASS_NAMES.index(cls)
            xmlbox = obj.find('bndbox')

            if xmlbox is None:
                continue

            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert_box((w, h), b)
            out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
            obj_count += 1

        in_file.close()
        out_file.close()
        return True

    except Exception as e:
        print(f"错误：处理 {xml_path} 时出错 - {str(e)}")
        return False


def process_dataset(voc_dir, output_base_dir):
    """处理整个数据集"""
    # 检查VOC目录是否存在
    if not os.path.exists(voc_dir):
        print(f"错误：VOC目录不存在 - {voc_dir}")
        print(f"当前工作目录：{os.getcwd()}")
        return

    sets = ['train', 'val', 'test']

    for image_set in sets:
        print(f"\n开始处理 {image_set} 集...")

        # 构建图片列表文件路径
        list_file = os.path.join(voc_dir, 'ImageSets', 'Main', f'{image_set}.txt')

        # 检查列表文件是否存在
        if not os.path.exists(list_file):
            print(f"警告：文件不存在 - {list_file}")
            continue

        # 读取图片ID列表
        with open(list_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines() if line.strip()]

        print(f"找到 {len(image_ids)} 张图片")

        # 创建输出目录
        label_dir = os.path.join(output_base_dir, 'labels', image_set)
        image_dir = os.path.join(output_base_dir, 'images', image_set)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # 计数器
        success_count = 0
        failed_count = 0

        # 转换标注并复制图片
        for i, image_id in enumerate(image_ids):
            if (i + 1) % 100 == 0:
                print(f"  处理进度: {i + 1}/{len(image_ids)}")

            # 转换标注
            if convert_annotation(voc_dir, image_id, label_dir):
                # 复制图片（尝试不同扩展名）
                img_copied = False
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    src_img = os.path.join(voc_dir, 'JPEGImages', f'{image_id}{ext}')
                    if os.path.exists(src_img):
                        dst_img = os.path.join(image_dir, f'{image_id}{ext}')
                        shutil.copy(src_img, dst_img)
                        img_copied = True
                        break

                if img_copied:
                    success_count += 1
                else:
                    print(f"警告：图片文件不存在 - {image_id}")
                    failed_count += 1
            else:
                failed_count += 1

        print(f"\n{image_set} 集处理完成：")
        print(f"  成功: {success_count} 张")
        print(f"  失败: {failed_count} 张")


# ================ 主程序 ================
if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前脚本目录: {current_dir}")

    # 方式1：使用绝对路径（根据你的实际路径修改）
    VOC_DIR = r"E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\VOC2007"

    # 方式2：使用相对路径（推荐）
    # 如果VOC2007在项目根目录下
    # VOC_DIR = os.path.join(current_dir, "..", "VOC2007")


    # 输出目录（相对于项目根目录）
    OUTPUT_DIR = os.path.join(current_dir, "..", "data")

    print(f"VOC数据路径: {VOC_DIR}")
    print(f"输出数据路径: {OUTPUT_DIR}")
    print(f"VOC目录是否存在: {os.path.exists(VOC_DIR)}")

    # 检查必要的子文件夹
    required_dirs = [
        os.path.join(VOC_DIR, 'Annotations'),
        os.path.join(VOC_DIR, 'JPEGImages'),
        os.path.join(VOC_DIR, 'ImageSets', 'Main')
    ]

    all_exist = True
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        print(f"  {dir_path}: {'✓ 存在' if exists else '✗ 不存在'}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n错误：VOC2007目录结构不完整！")
        print("请确保包含以下文件夹：")
        print("  - Annotations/")
        print("  - JPEGImages/")
        print("  - ImageSets/Main/")
    else:
        print("\n开始转换数据集...")
        process_dataset(VOC_DIR, OUTPUT_DIR)
        print("\n数据集转换完成！")