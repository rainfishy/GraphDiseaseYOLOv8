import os


def find_voc_directory(start_path="."):
    """在项目中查找VOC2007文件夹"""
    print(f"从 {os.path.abspath(start_path)} 开始搜索...\n")

    for root, dirs, files in os.walk(start_path):
        if 'VOC2007' in dirs:
            voc_path = os.path.join(root, 'VOC2007')
            print(f"✓ 找到VOC2007:")
            print(f"  路径: {os.path.abspath(voc_path)}")

            # 检查子文件夹
            annotations = os.path.join(voc_path, 'Annotations')
            images = os.path.join(voc_path, 'JPEGImages')
            imagesets = os.path.join(voc_path, 'ImageSets', 'Main')

            print(f"\n  子文件夹检查:")
            print(f"    Annotations: {'✓' if os.path.exists(annotations) else '✗'}")
            print(f"    JPEGImages: {'✓' if os.path.exists(images) else '✗'}")
            print(f"    ImageSets/Main: {'✓' if os.path.exists(imagesets) else '✗'}")

            # 给出相对路径建议
            rel_path = os.path.relpath(voc_path, os.path.join(start_path, 'scripts'))
            print(f"\n  在voc_to_yolo.py中使用:")
            print(f"    VOC_DIR = r'{os.path.abspath(voc_path)}'")
            print(f"  或使用相对路径:")
            print(f"    VOC_DIR = os.path.join(current_dir, '..', '{rel_path.replace(os.sep, '/')}')")
            print("")


# 从项目根目录开始搜索
find_voc_directory("..")