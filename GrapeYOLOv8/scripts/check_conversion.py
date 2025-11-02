import os

# 检查文件数量
splits = ['train', 'val', 'test']
base_dir = '../data'

print("=" * 50)
print("检查数据转换结果")
print("=" * 50)

for split in splits:
    img_dir = os.path.join(base_dir, 'images', split)
    label_dir = os.path.join(base_dir, 'labels', split)

    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"\n{split} 集: 目录不存在！")
        continue

    img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    label_count = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    print(f"\n{split} 集:")
    print(f"  图片数量: {img_count}")
    print(f"  标签数量: {label_count}")
    print(f"  匹配状态: {'✅ 正常' if img_count == label_count else '❌ 不匹配'}")

print("\n" + "=" * 50)

# 检查标签内容格式
print("\n查看标签文件示例 (train/第一个):")
print("=" * 50)

train_label_dir = os.path.join(base_dir, 'labels', 'train')
first_label = sorted(os.listdir(train_label_dir))[0]
label_path = os.path.join(train_label_dir, first_label)

print(f"文件名: {first_label}")
with open(label_path, 'r') as f:
    lines = f.readlines()[:3]  # 只显示前3行
    for i, line in enumerate(lines, 1):
        print(f"  行{i}: {line.strip()}")

print("\n格式说明: <类别ID> <x_center> <y_center> <width> <height>")
print("所有值都归一化到 0-1 范围")
print("=" * 50)