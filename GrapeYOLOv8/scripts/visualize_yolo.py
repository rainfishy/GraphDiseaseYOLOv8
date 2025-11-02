import cv2
import matplotlib.pyplot as plt
import random
import os

CLASS_NAMES = ['black_rot', 'blight', 'black_measles', 'Healthy']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


def visualize_sample(image_path, label_path):
    """可视化YOLO格式标注"""
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]

    # 读取标签
    if not os.path.exists(label_path):
        print(f"标签文件不存在: {label_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 绘制每个边界框
    for line in lines:
        data = line.strip().split()
        if len(data) < 5:
            continue

        cls_id = int(data[0])
        x_center, y_center, width, height = map(float, data[1:])

        # 转换为像素坐标
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # 绘制边界框
        color = COLORS[cls_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 添加类别标签
        label_text = CLASS_NAMES[cls_id]
        cv2.putText(img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 显示
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Sample: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 随机选择3张训练集图片可视化
data_dir = '../data'
train_img_dir = os.path.join(data_dir, 'images', 'train')
train_label_dir = os.path.join(data_dir, 'labels', 'train')

img_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))]

print(f"找到 {len(img_files)} 张训练图片，随机可视化3张...")

for i in range(min(3, len(img_files))):
    img_name = random.choice(img_files)
    img_path = os.path.join(train_img_dir, img_name)
    label_path = os.path.join(train_label_dir, img_name.rsplit('.', 1)[0] + '.txt')

    print(f"\n可视化 {i + 1}/3: {img_name}")
    visualize_sample(img_path, label_path)