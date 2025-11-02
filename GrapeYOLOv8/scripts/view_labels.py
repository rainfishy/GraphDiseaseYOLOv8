import os
import random

CLASS_NAMES = ['black_rot', 'blight', 'black_measles', 'Healthy']


def view_sample_labels():
    """查看标签文件示例"""

    label_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'labels', 'train')
    label_dir = os.path.abspath(label_dir)

    if not os.path.exists(label_dir):
        print(f"错误: 目录不存在 - {label_dir}")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    if not label_files:
        print("错误: 没有找到标签文件")
        return

    print("=" * 70)
    print("标签文件内容示例")
    print("=" * 70)

    # 随机选择3个标签文件
    samples = random.sample(label_files, min(3, len(label_files)))

    for i, label_file in enumerate(samples, 1):
        label_path = os.path.join(label_dir, label_file)

        print(f"\n[示例 {i}] 文件名: {label_file}")
        print("-" * 70)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print("  (空文件 - 可能是无标注图片)")
            continue

        print(f"  标注对象数: {len(lines)}")
        print(f"\n  内容:")

        for j, line in enumerate(lines[:5], 1):  # 只显示前5行
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                print(f"    对象{j}: {CLASS_NAMES[cls_id]:15s} "
                      f"[x={x_center:.4f}, y={y_center:.4f}, "
                      f"w={width:.4f}, h={height:.4f}]")

        if len(lines) > 5:
            print(f"    ... (还有 {len(lines) - 5} 个对象)")

    print("\n" + "=" * 70)
    print("标签格式说明:")
    print("  <类别ID> <x中心> <y中心> <宽度> <高度>")
    print("  所有坐标值都归一化到 0-1 范围")
    print("=" * 70)


if __name__ == "__main__":
    view_sample_labels()