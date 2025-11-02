import os
import matplotlib.pyplot as plt
from collections import Counter

CLASS_NAMES = ['black_rot', 'blight', 'black_measles', 'Healthy']


def count_class_distribution(label_dir):
    """统计类别分布"""

    class_counter = Counter()
    total_objects = 0
    empty_files = 0

    if not os.path.exists(label_dir):
        return class_counter, total_objects, empty_files

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                class_counter[cls_id] += 1
                total_objects += 1

    return class_counter, total_objects, empty_files


def analyze_dataset():
    """分析整个数据集"""

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    splits = ['train', 'val', 'test']

    print("=" * 70)
    print("数据集统计分析")
    print("=" * 70)

    all_stats = {}

    for split in splits:
        label_dir = os.path.join(base_dir, 'labels', split)
        class_counter, total_objects, empty_files = count_class_distribution(label_dir)

        print(f"\n【{split.upper()} 集】")
        print("-" * 70)

        if not class_counter:
            print("  ❌ 没有数据")
            continue

        # 打印类别分布
        print(f"  总标注对象数: {total_objects}")
        print(f"  空标签文件数: {empty_files}")
        print(f"\n  各类别分布:")

        for cls_id in range(len(CLASS_NAMES)):
            count = class_counter.get(cls_id, 0)
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"    {CLASS_NAMES[cls_id]:15s}: {count:5d} ({percentage:5.2f}%)")

        all_stats[split] = class_counter

    # 绘制统计图
    print("\n" + "=" * 70)
    print("生成可视化图表...")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, split in enumerate(splits):
        if split not in all_stats or not all_stats[split]:
            continue

        class_counter = all_stats[split]

        # 准备数据
        classes = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
        counts = [class_counter.get(i, 0) for i in range(len(CLASS_NAMES))]

        # 绘制柱状图
        ax = axes[idx]
        bars = ax.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        ax.set_title(f'{split.upper()} Set', fontsize=14, fontweight='bold')
        ax.set_xlabel('Disease Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(base_dir, '..', 'runs')
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, 'dataset_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 图表已保存到: {save_path}")

    plt.show()


if __name__ == "__main__":
    analyze_dataset()