"""
数据增强结果验证脚本
"""

import os
import random
import cv2
import matplotlib.pyplot as plt
from collections import Counter

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置微软雅黑字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

CLASS_NAMES = ['black_rot', 'blight', 'black_measles', 'Healthy']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


def load_yolo_boxes(txt_path):
    """读取YOLO标注框"""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
                boxes.append((cls_id, bbox))
    return boxes


def draw_boxes(image, boxes):
    """在图像上绘制边界框"""
    h, w = image.shape[:2]

    for cls_id, bbox in boxes:
        x_center, y_center, box_w, box_h = bbox

        # 转换为像素坐标
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # 边界检查
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # 绘制框
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = CLASS_NAMES[cls_id]
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 5),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image


def verify_augmentation():
    """验证数据增强结果"""

    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data_augmented')

    print("=" * 70)
    print("🔍 数据增强结果验证")
    print("=" * 70)

    # 检查目录
    if not os.path.exists(base_dir):
        print("❌ data_augmented 目录不存在！")
        print("请先运行数据增强脚本")
        return

    # 统计各集合的图片数量
    print("\n📊 数据集规模统计:")
    print("-" * 70)

    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(base_dir, 'images', split)
        label_dir = os.path.join(base_dir, 'labels', split)

        if os.path.exists(img_dir):
            img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
            label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

            original = [f for f in img_files if '_aug' not in f]
            augmented = [f for f in img_files if '_aug' in f]

            print(f"{split:6s} | 原始: {len(original):4d} | "
                  f"增强: {len(augmented):4d} | 总计: {len(img_files):4d} | "
                  f"标签: {len(label_files):4d}")

    # 统计类别分布
    print("\n📈 训练集类别分布:")
    print("-" * 70)

    train_label_dir = os.path.join(base_dir, 'labels', 'train')
    class_counter = Counter()

    for label_file in os.listdir(train_label_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(train_label_dir, label_file)
        boxes = load_yolo_boxes(label_path)
        for cls_id, _ in boxes:
            class_counter[cls_id] += 1

    total_boxes = sum(class_counter.values())
    for cls_id in range(len(CLASS_NAMES)):
        count = class_counter.get(cls_id, 0)
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"{CLASS_NAMES[cls_id]:15s}: {count:5d} ({percentage:5.2f}%)")

    print(f"{'总计':15s}: {total_boxes:5d}")

    # 可视化对比：原始图 vs 增强图
    print("\n🖼️  生成可视化对比图...")
    print("-" * 70)

    train_img_dir = os.path.join(base_dir, 'images', 'train')
    train_label_dir = os.path.join(base_dir, 'labels', 'train')

    # 选择一个有增强样本的原始图片
    all_files = os.listdir(train_img_dir)
    original_files = [f for f in all_files if '_aug' not in f and f.endswith('.jpg')]

    if not original_files:
        print("❌ 没有找到原始图片")
        return

    # 随机选择一张
    sample_base = random.choice([f.replace('.jpg', '') for f in original_files])

    # 查找对应的增强图片
    related_files = [sample_base + '.jpg']
    for i in range(10):  # 最多找10个增强样本
        aug_file = f"{sample_base}_aug{i}.jpg"
        if aug_file in all_files:
            related_files.append(aug_file)

    # 创建对比图
    n_images = min(4, len(related_files))
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))

    if n_images == 1:
        axes = [axes]

    for i, img_file in enumerate(related_files[:n_images]):
        img_path = os.path.join(train_img_dir, img_file)
        label_path = os.path.join(train_label_dir, img_file.replace('.jpg', '.txt'))

        # 读取图片
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取并绘制边界框
        boxes = load_yolo_boxes(label_path)
        img = draw_boxes(img, boxes)

        # 显示
        axes[i].imshow(img)
        title = "原始图片" if i == 0 else f"增强样本 {i}"
        axes[i].set_title(f"{title}\n病害数: {len(boxes)}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()

    # 保存对比图
    save_path = os.path.join(base_dir, 'augmentation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图已保存: {save_path}")

    plt.show()

    # 生成类别分布柱状图
    print("\n📊 生成类别分布图...")

    fig, ax = plt.subplots(figsize=(10, 6))

    classes = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    counts = [class_counter.get(i, 0) for i in range(len(CLASS_NAMES))]
    colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    bars = ax.bar(classes, counts, color=colors_bar)
    ax.set_title('训练集类别分布', fontsize=14, fontweight='bold')
    ax.set_xlabel('病害类别', fontsize=12)
    ax.set_ylabel('数量', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(base_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 分布图已保存: {save_path}")

    plt.show()

    print("\n" + "=" * 70)
    print("✅ 验证完成！")
    print("=" * 70)


if __name__ == "__main__":
    verify_augmentation()