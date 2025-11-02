import xml.etree.ElementTree as ET
import os
from collections import Counter


def extract_all_classes(voc_dir):
    """
    扫描所有XML文件，提取所有类别名称
    """
    annotations_dir = os.path.join(voc_dir, 'Annotations')

    if not os.path.exists(annotations_dir):
        print(f"错误：目录 {annotations_dir} 不存在！")
        return

    # 统计所有类别
    class_counter = Counter()

    # 遍历所有XML文件
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]

    print(f"正在扫描 {len(xml_files)} 个标注文件...\n")

    for xml_file in xml_files:
        xml_path = os.path.join(annotations_dir, xml_file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 提取所有object的name
            for obj in root.iter('object'):
                name = obj.find('name')
                if name is not None:
                    class_name = name.text
                    class_counter[class_name] += 1

        except Exception as e:
            print(f"警告：解析 {xml_file} 时出错: {e}")

    # 打印结果
    print("=" * 50)
    print("数据集中的所有类别：")
    print("=" * 50)

    for i, (class_name, count) in enumerate(class_counter.most_common()):
        print(f"{i}. '{class_name}' - 出现 {count} 次")

    print("\n" + "=" * 50)
    print("请将以下代码复制到 voc_to_yolo.py 中：")
    print("=" * 50)

    # 生成CLASS_NAMES列表
    class_list = [f"'{name}'" for name, _ in class_counter.most_common()]
    print(f"CLASS_NAMES = [{', '.join(class_list)}]")
    print("=" * 50)


# 运行
VOC_DIR = "E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\VOC2007"  # 修改为你的VOC2007路径
extract_all_classes(VOC_DIR)