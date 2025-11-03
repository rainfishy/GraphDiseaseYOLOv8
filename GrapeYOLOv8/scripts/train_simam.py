"""
YOLOv8n + SimAM 训练脚本
葡萄叶片病害检测 - 改进模型
"""

import os
from ultralytics import YOLO
import torch


def train_simam_model():
    print("=" * 70)
    print("🚀 开始训练 YOLOv8n + SimAM 改进模型")
    print("=" * 70)

    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("\n⚠️ 未检测到GPU，将使用CPU训练（速度较慢）")

    # 训练配置（结合baseline的成功参数）
    config = {
        # 数据配置
        'data': os.path.join('..', 'data_augmented', 'grape_augmented.yaml'),

        # 训练轮数（和baseline保持一致）
        'epochs': 150,

        # 图像和批次大小
        'imgsz': 640,
        'batch': 16,
        'workers': 4,

        # 早停策略
        'patience': 20,

        # 学习率（和baseline一致）
        'lr0': 0.01,  # 初始学习率
        'lrf': 0.01,  # 最终学习率因子

        # 优化器参数（和baseline一致）
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # 保存设置
        'save': True,
        'save_period': 10,  # 每10轮保存一次

        # 输出路径
        'project': os.path.join('..', 'runs'),
        'name': 'train_simam',
        'exist_ok': True,

        # 其他设置
        'plots': True,  # 生成训练图表
        'verbose': True,  # 详细输出
    }

    print("\n📋 训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    try:
        # 加载改进模型配置
        print("\n🔄 加载YOLOv8n + SimAM模型配置...")
        model_yaml = os.path.join('..', 'models', 'yolov8n_simam.yaml')

        if not os.path.exists(model_yaml):
            print(f"❌ 未找到模型配置文件: {model_yaml}")
            return None

        model = YOLO(model_yaml)
        print("✅ 模型配置加载成功!")

        # 开始训练
        print("\n" + "=" * 70)
        print("🔥 开始训练...")
        print("=" * 70 + "\n")

        results = model.train(**config)

        print("\n" + "=" * 70)
        print("✅ SimAM改进模型训练完成!")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_simam():
    """验证训练好的SimAM模型"""
    print("\n" + "=" * 70)
    print("📊 验证 SimAM 改进模型...")
    print("=" * 70)

    model_path = os.path.join('..', 'runs', 'train_simam', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f"❌ 未找到权重文件: {model_path}")
        print("   请先完成训练!")
        return None

    try:
        model = YOLO(model_path)
        results = model.val(
            data=os.path.join('..', 'data_augmented', 'grape_augmented.yaml'),
            split='val'
        )

        print("\n📈 SimAM改进模型性能:")
        print(f"   mAP@0.5: {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"   精确率 (Precision): {results.box.mp:.4f}")
        print(f"   召回率 (Recall): {results.box.mr:.4f}")

        return results

    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_baseline():
    """对比baseline和SimAM模型的性能"""
    print("\n" + "=" * 70)
    print("📊 对比分析: Baseline vs SimAM")
    print("=" * 70)

    baseline_path = os.path.join('..', 'runs', 'baseline_yolov8n', 'weights', 'best.pt')
    simam_path = os.path.join('..', 'runs', 'train_simam', 'weights', 'best.pt')

    if not os.path.exists(baseline_path):
        print("⚠️ 未找到baseline模型，跳过对比")
        return

    if not os.path.exists(simam_path):
        print("⚠️ 未找到SimAM模型，跳过对比")
        return

    try:
        # 加载两个模型
        baseline = YOLO(baseline_path)
        simam = YOLO(simam_path)

        data_yaml = os.path.join('..', 'data_augmented', 'grape_augmented.yaml')

        # 验证baseline
        print("\n🔍 验证 Baseline 模型...")
        baseline_results = baseline.val(data=data_yaml, split='test')

        # 验证SimAM
        print("\n🔍 验证 SimAM 模型...")
        simam_results = simam.val(data=data_yaml, split='test')

        # 对比结果
        print("\n" + "=" * 70)
        print("📊 性能对比 (测试集)")
        print("=" * 70)

        metrics = [
            ('mAP@0.5', 'map50'),
            ('mAP@0.5:0.95', 'map'),
            ('Precision', 'mp'),
            ('Recall', 'mr')
        ]

        print(f"\n{'指标':<20} {'Baseline':<15} {'SimAM':<15} {'提升':<15}")
        print("-" * 70)

        for metric_name, metric_key in metrics:
            baseline_val = getattr(baseline_results.box, metric_key)
            simam_val = getattr(simam_results.box, metric_key)
            improvement = ((simam_val - baseline_val) / baseline_val) * 100

            print(f"{metric_name:<20} {baseline_val:<15.4f} {simam_val:<15.4f} {improvement:>+6.2f}%")

        print("=" * 70)

    except Exception as e:
        print(f"❌ 对比过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 训练SimAM改进模型
    train_results = train_simam_model()

    if train_results:
        # 验证模型
        val_results = validate_simam()

        if val_results:
            # 与baseline对比
            compare_with_baseline()

            print("\n" + "=" * 70)
            print("🎉 所有任务完成!")
            print("=" * 70)
    else:
        print("\n❌ 训练失败，请检查错误信息")