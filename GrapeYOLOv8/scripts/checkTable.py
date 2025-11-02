import os
import xml.etree.ElementTree as ET
from collections import Counter

annotations_dir = 'E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\VOC2007\Annotations'
all_labels = []

for filename in os.listdir(annotations_dir):
    if filename.endswith('.xml'):
        file_path = os.path.join(annotations_dir, filename)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                label = obj.find('name').text
                all_labels.append(label)
        except Exception as e:
            print(f"解析文件 {filename} 时出错: {e}")

label_counts = Counter(all_labels)
print("所有类别及其出现次数：")
for label, count in label_counts.most_common():
    print(f"'{label}': {count} 次")

# 打印所有类别，看看是否有拼写错误
print("\n所有唯一的类别标签：")
for label in sorted(label_counts.keys()):
    print(f"'{label}'")