"""
数据增强脚本 - 改进版B（推荐）
针对葡萄叶片小目标病害检测优化

作者：实验组
版本：2.0
日期：2025
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 类别定义
CLASS_NAMES = ['black_rot', 'blight', 'black_measles', 'Healthy']


# ============================================================================
# 数据增强策略定义
# ============================================================================

def get_augmentation_pipeline(version='B'):
    """
    获取数据增强管道

    参数:
        version: 'A' (保守), 'B' (推荐), 'C' (激进)

    返回:
        albumentations.Compose对象
    """

    if version == 'A':
        # 版本A：保守版（接近原实验报告）
        pipeline = A.Compose([
            # 1. 随机擦除（原始设置）
            A.CoarseDropout(
                max_holes=5,
                max_height=0.1,
                max_width=0.1,
                fill_value=0,
                p=0.5
            ),

            # 2. HSV色域扩展
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.7
            ),

            # 3. 几何变换
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            ], p=0.6),

            # 4. 亮度对比度
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),

            # 5. 模糊和噪声
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))

    elif version == 'B':
        # 版本B：改进版（推荐使用）
        pipeline = A.Compose([
            # 1. 细粒度随机擦除（避免完全遮挡小目标）
            A.CoarseDropout(
                max_holes=20,  # 更多小孔洞
                max_height=0.04,  # 减小单个孔洞尺寸
                max_width=0.04,
                min_height=0.015,  # 设置最小尺寸
                min_width=0.015,
                fill_value=0,
                p=0.6  # 提高应用概率
            ),

            # 2. HSV色域扩展（增强版）
            A.HueSaturationValue(
                hue_shift_limit=12,
                sat_shift_limit=40,  # 增大饱和度变化
                val_shift_limit=30,  # 增大明度变化
                p=0.8  # 提高应用概率
            ),

            # 3. 几何变换（增强版）
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
            ], p=0.7),

            # 4. 光照变化（增强版）
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),

            # 5. 模糊和噪声（增强版）
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.4),

            # 6. 天气条件模拟（新增）
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=10,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=3,
                    brightness_coefficient=0.9,
                    rain_type='drizzle',
                    p=1.0
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=1.0
                ),
            ], p=0.2),

        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.4  # 提高最小可见度
        ))

    elif version == 'C':
        # 版本C：激进版（最大增强强度）
        pipeline = A.Compose([
            # 1. 极细粒度随机擦除
            A.CoarseDropout(
                max_holes=30,
                max_height=0.03,
                max_width=0.03,
                min_height=0.01,
                min_width=0.01,
                fill_value=0,
                p=0.7
            ),

            # 2. 极强HSV变化
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=50,
                val_shift_limit=40,
                p=0.9
            ),

            # 3. 复杂几何变换
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
                A.Perspective(scale=(0.05, 0.1), p=1.0),
            ], p=0.8),

            # 4. 极强光照变化
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.8
            ),

            # 5. 复杂模糊和噪声
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=(3, 9), p=1.0),
                A.MedianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(20.0, 80.0), p=1.0),
            ], p=0.5),

            # 6. 多种天气条件
            A.OneOf([
                A.RandomRain(drop_length=15, drop_width=2, p=1.0),
                A.RandomShadow(num_shadows_upper=3, p=1.0),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
            ], p=0.3),

        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.5
        ))

    else:
        raise ValueError(f"未知的版本: {version}。请选择 'A', 'B' 或 'C'")

    return pipeline


# ============================================================================
# 辅助函数
# ============================================================================

def load_yolo_annotation(txt_path):
    """读取YOLO格式标注"""
    bboxes = []
    class_labels = []

    if not os.path.exists(txt_path):
        return bboxes, class_labels

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:]))

                # 验证bbox合法性
                if all(0 <= x <= 1 for x in bbox):
                    class_labels.append(class_id)
                    bboxes.append(bbox)

    return bboxes, class_labels


def save_yolo_annotation(txt_path, bboxes, class_labels):
    """保存YOLO格式标注"""
    with open(txt_path, 'w') as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            line = f"{cls_id} {' '.join([f'{x:.6f}' for x in bbox])}\n"
            f.write(line)


def augment_single_image(img_path, label_path, output_img_dir, output_label_dir,
                         aug_pipeline, aug_count=3, base_name=None):
    """
    对单张图片进行数据增强

    参数:
        img_path: 原始图片路径
        label_path: 原始标注路径
        output_img_dir: 输出图片目录
        output_label_dir: 输出标注目录
        aug_pipeline: 增强管道
        aug_count: 每张图片生成的增强样本数量
        base_name: 基础文件名

    返回:
        成功生成的增强样本数量
    """

    # 读取图片
    image = cv2.imread(img_path)
    if image is None:
        return 0

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image.shape[:2]

    # 读取标注
    bboxes, class_labels = load_yolo_annotation(label_path)

    if base_name is None:
        base_name = Path(img_path).stem

    success_count = 0

    # 生成多个增强样本
    for i in range(aug_count):
        try:
            # 应用增强
            if len(bboxes) == 0:
                # 没有标注框的图片（可能是健康叶片）
                augmented = aug_pipeline(image=image, bboxes=[], class_labels=[])
                aug_image = augmented['image']
                aug_bboxes = []
                aug_labels = []
            else:
                augmented = aug_pipeline(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )

                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']

                # 检查增强后是否还有有效的边界框
                if len(aug_bboxes) == 0:
                    continue  # 所有框都被裁掉了，跳过这个增强

            # 保存增强后的图片
            aug_img_name = f"{base_name}_aug{i}.jpg"
            aug_img_path = os.path.join(output_img_dir, aug_img_name)
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(aug_img_path, aug_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 保存增强后的标注
            aug_label_name = f"{base_name}_aug{i}.txt"
            aug_label_path = os.path.join(output_label_dir, aug_label_name)
            save_yolo_annotation(aug_label_path, aug_bboxes, aug_labels)

            success_count += 1

        except Exception as e:
            # 静默失败，继续下一个增强
            continue

    return success_count


# ============================================================================
# 主增强函数
# ============================================================================

def augment_dataset(input_base_dir, output_base_dir, aug_per_image=3,
                    version='B', only_train=True):
    """
    对整个数据集进行增强

    参数:
        input_base_dir: 输入数据集目录
        output_base_dir: 输出数据集目录
        aug_per_image: 每张图片生成的增强样本数
        version: 增强版本 ('A', 'B', 'C')
        only_train: 是否只增强训练集
    """

    print("=" * 70)
    print("🚀 葡萄叶病害数据增强系统")
    print("=" * 70)
    print(f"📌 版本: {version}")
    print(f"📌 输入目录: {os.path.abspath(input_base_dir)}")
    print(f"📌 输出目录: {os.path.abspath(output_base_dir)}")
    print(f"📌 增强倍数: 每张图 × {aug_per_image}")
    print("=" * 70)

    # 检查输入目录
    if not os.path.exists(input_base_dir):
        print(f"\n❌ 错误: 输入目录不存在！")
        return

    # 获取增强管道
    aug_pipeline = get_augmentation_pipeline(version)

    # 确定要处理的数据集
    splits = ['train'] if only_train else ['train', 'val']

    # 统计信息
    total_stats = {}

    for split in splits:
        print(f"\n{'=' * 70}")
        print(f"📊 处理 {split.upper()} 集")
        print(f"{'=' * 70}")

        # 输入路径
        input_img_dir = os.path.join(input_base_dir, 'images', split)
        input_label_dir = os.path.join(input_base_dir, 'labels', split)

        # 检查输入目录
        if not os.path.exists(input_img_dir):
            print(f"❌ 跳过: 目录不存在 - {input_img_dir}")
            continue

        # 输出路径
        output_img_dir = os.path.join(output_base_dir, 'images', split)
        output_label_dir = os.path.join(output_base_dir, 'labels', split)

        # 创建输出目录
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        # 获取所有图片
        img_files = sorted([f for f in os.listdir(input_img_dir)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        print(f"\n📁 原始图片数: {len(img_files)}")

        # 步骤1: 复制原始数据
        print(f"\n[步骤 1/2] 复制原始数据...")
        for img_file in tqdm(img_files, desc="  复制进度", ncols=70):
            # 复制图片
            src_img = os.path.join(input_img_dir, img_file)
            dst_img = os.path.join(output_img_dir, img_file)
            shutil.copy2(src_img, dst_img)

            # 复制标签
            label_file = Path(img_file).stem + '.txt'
            src_label = os.path.join(input_label_dir, label_file)
            dst_label = os.path.join(output_label_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

        # 步骤2: 生成增强数据
        print(f"\n[步骤 2/2] 生成增强数据 (版本{version})...")
        total_augmented = 0
        failed_count = 0

        for img_file in tqdm(img_files, desc="  增强进度", ncols=70):
            img_path = os.path.join(input_img_dir, img_file)
            label_file = Path(img_file).stem + '.txt'
            label_path = os.path.join(input_label_dir, label_file)

            count = augment_single_image(
                img_path, label_path,
                output_img_dir, output_label_dir,
                aug_pipeline, aug_per_image,
                base_name=Path(img_file).stem
            )

            total_augmented += count
            if count < aug_per_image:
                failed_count += 1

        # 统计最终结果
        final_img_count = len([f for f in os.listdir(output_img_dir)
                               if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        final_label_count = len([f for f in os.listdir(output_label_dir)
                                 if f.endswith('.txt')])

        # 保存统计信息
        total_stats[split] = {
            'original': len(img_files),
            'augmented': total_augmented,
            'final': final_img_count,
            'failed': failed_count
        }

        # 打印结果
        print(f"\n✅ {split.upper()} 集处理完成:")
        print(f"   原始图片: {len(img_files):5d} 张")
        print(f"   增强生成: {total_augmented:5d} 张")
        print(f"   增强失败: {failed_count:5d} 张")
        print(f"   图片总数: {final_img_count:5d} 张")
        print(f"   标签总数: {final_label_count:5d} 个")
        print(f"   扩增倍数: {final_img_count / len(img_files):.2f}x")

    # 复制测试集（不增强）
    print(f"\n{'=' * 70}")
    print("📋 复制 TEST 集（不进行增强）")
    print(f"{'=' * 70}")

    test_copied = False
    for subdir in ['images', 'labels']:
        src_dir = os.path.join(input_base_dir, subdir, 'test')
        dst_dir = os.path.join(output_base_dir, subdir, 'test')

        if os.path.exists(src_dir):
            os.makedirs(dst_dir, exist_ok=True)
            files = os.listdir(src_dir)
            for file in tqdm(files, desc=f"  复制{subdir}", ncols=70):
                shutil.copy2(
                    os.path.join(src_dir, file),
                    os.path.join(dst_dir, file)
                )
            print(f"✅ {subdir}/test: {len(files):4d} 个文件")
            test_copied = True

    if test_copied:
        test_img_count = len(os.listdir(os.path.join(output_base_dir, 'images', 'test')))
        total_stats['test'] = {
            'original': test_img_count,
            'augmented': 0,
            'final': test_img_count,
            'failed': 0
        }

    # 打印最终统计
    print(f"\n{'=' * 70}")
    print("🎉 数据增强全部完成！")
    print(f"{'=' * 70}")
    print(f"\n📊 最终统计:")
    print(f"{'=' * 70}")

    for split, stats in total_stats.items():
        print(f"{split.upper():6s} | 原始: {stats['original']:4d} | "
              f"最终: {stats['final']:4d} | 扩增: {stats['final'] / stats['original']:.2f}x")

    total_final = sum(stats['final'] for stats in total_stats.values())
    total_original = sum(stats['original'] for stats in total_stats.values())

    print(f"{'=' * 70}")
    print(f"总计   | 原始: {total_original:4d} | 最终: {total_final:4d} | "
          f"扩增: {total_final / total_original:.2f}x")
    print(f"{'=' * 70}")
    print(f"\n📂 增强后数据保存在: {os.path.abspath(output_base_dir)}")
    print(f"{'=' * 70}\n")


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='葡萄叶病害数据增强')
    parser.add_argument('--version', type=str, default='B',
                        choices=['A', 'B', 'C'],
                        help='增强版本: A(保守), B(推荐), C(激进)')
    parser.add_argument('--aug-count', type=int, default=3,
                        help='每张图片生成的增强样本数 (默认: 3)')
    parser.add_argument('--only-train', action='store_true', default=True,
                        help='仅增强训练集')

    args = parser.parse_args()

    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(current_dir, '..', 'data')
    OUTPUT_DIR = os.path.join(current_dir, '..', 'data_augmented')

    # 执行增强
    augment_dataset(
        INPUT_DIR,
        OUTPUT_DIR,
        aug_per_image=args.aug_count,
        version=args.version,
        only_train=args.only_train
    )