"""
最终验证清单
"""

import os


def final_check():
    """检查所有必要文件是否就绪"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(current_dir, '..')

    print("=" * 70)
    print("🔍 最终验证清单")
    print("=" * 70)

    checks = []

    # 检查1: 原始数据
    data_dir = os.path.join(project_dir, 'data')
    checks.append(("原始数据目录", os.path.exists(data_dir)))

    # 检查2: 增强数据
    aug_dir = os.path.join(project_dir, 'data_augmented')
    checks.append(("增强数据目录", os.path.exists(aug_dir)))

    # 检查3: 训练集
    train_img = os.path.join(aug_dir, 'images', 'train')
    train_label = os.path.join(aug_dir, 'labels', 'train')
    checks.append(("训练集图片", os.path.exists(train_img)))
    checks.append(("训练集标签", os.path.exists(train_label)))

    # 检查4: 验证集
    val_img = os.path.join(aug_dir, 'images', 'val')
    val_label = os.path.join(aug_dir, 'labels', 'val')
    checks.append(("验证集图片", os.path.exists(val_img)))
    checks.append(("验证集标签", os.path.exists(val_label)))

    # 检查5: 测试集
    test_img = os.path.join(aug_dir, 'images', 'test')
    test_label = os.path.join(aug_dir, 'labels', 'test')
    checks.append(("测试集图片", os.path.exists(test_img)))
    checks.append(("测试集标签", os.path.exists(test_label)))

    # 检查6: YAML配置
    yaml_file = os.path.join(aug_dir, 'grape_augmented.yaml')
    checks.append(("YAML配置文件", os.path.exists(yaml_file)))

    # 打印结果
    print("\n检查结果:")
    print("-" * 70)

    all_pass = True
    for item, status in checks:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {item:20s}: {'通过' if status else '失败'}")
        if not status:
            all_pass = False

    # 统计数据
    if all_pass:
        print("\n数据统计:")
        print("-" * 70)

        for split in ['train', 'val', 'test']:
            img_dir = os.path.join(aug_dir, 'images', split)
            label_dir = os.path.join(aug_dir, 'labels', split)

            if os.path.exists(img_dir):
                img_count = len([f for f in os.listdir(img_dir)
                                 if f.endswith('.jpg')])
                label_count = len([f for f in os.listdir(label_dir)
                                   if f.endswith('.txt')])

                print(f"{split:6s}: 图片 {img_count:5d} 张, 标签 {label_count:5d} 个")

    print("\n" + "=" * 70)

    if all_pass:
        print("🎉 所有检查通过！数据准备完成！")
        print("\n下一步：")
        print("  1. 安装 YOLOv8 环境")
        print("  2. 训练基线模型")
        print("  3. 开始模型改进")
    else:
        print("❌ 部分检查未通过，请检查上述失败项")

    print("=" * 70)


if __name__ == "__main__":
    final_check()