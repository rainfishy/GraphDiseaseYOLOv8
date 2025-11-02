import os
import xml.etree.ElementTree as ET

# 设置Annotations文件夹路径
annotations_dir = 'E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\VOC2007\Annotations'

# 定义修正映射 - 将错误的标签映射到正确的标签
correction_map = {
    'blck_measles': 'black_measles',  # 修正拼写错误
}

# 统计修正情况
correction_count = 0
modified_files = []

print("开始检查并修正标签...")

# 遍历Annotations文件夹中的所有XML文件
for filename in os.listdir(annotations_dir):
    if filename.endswith('.xml'):
        file_path = os.path.join(annotations_dir, filename)

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            file_modified = False

            # 检查所有object标签
            for obj in root.findall('object'):
                label_tag = obj.find('name')
                old_label = label_tag.text

                # 如果找到需要修正的标签
                if old_label in correction_map:
                    new_label = correction_map[old_label]
                    print(f"在文件 {filename} 中将 '{old_label}' 修正为 '{new_label}'")
                    label_tag.text = new_label
                    file_modified = True
                    correction_count += 1

            # 如果文件被修改，保存更改
            if file_modified:
                tree.write(file_path, encoding='utf-8', xml_declaration=True)
                modified_files.append(filename)

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

# 输出修正总结
print(f"\n修正完成！")
print(f"总共修正了 {correction_count} 个标签")
print(f"修改了 {len(modified_files)} 个文件")

if modified_files:
    print("被修改的文件：")
    for file in modified_files:
        print(f"  - {file}")
else:
    print("没有找到需要修正的标签。")