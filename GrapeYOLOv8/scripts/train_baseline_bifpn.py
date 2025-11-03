"""
训练 Baseline + BiFPN 模型
"""
import os
from ultralytics import YOLO
import torch


def train_baseline_bifpn():
    """训练YOLOv8n+BiFPN"""

    print("=" * 70)
    print("🚀 训练 YOLOv8n + BiFPN 模型")
    print("=" * 70)

    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("\n⚠️ 未检测到GPU")

    # 训练配置
    config = {
        'data': os.path.join('..', 'data_augmented', 'grape_augmented.yaml'),
        'epochs': 150,
        'imgsz': 640,
        'batch': 16,
        'workers': 4,
        'patience': 20,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'save': True,
        'save_period': 10,
        'project': os.path.join('..', 'runs'),
        'name': 'train_bifpn',
        'exist_ok': True,
        'plots': True,
        'verbose': True,
    }

    print("\n📋 训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    try:
        print("\n🔄 加载模型...")
        model_yaml = os.path.join('..', 'models', 'yolov8n_bifpn.yaml')

        if not os.path.exists(model_yaml):
            print(f"❌ 未找到模型配置: {model_yaml}")
            return None

        model = YOLO(model_yaml)
        print("✅ 模型加载成功!")

        print("\n" + "=" * 70)
        print("🔥 开始训练...")
        print("=" * 70 + "\n")

        results = model.train(**config)

        print("\n" + "=" * 70)
        print("✅ BiFPN模型训练完成!")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_bifpn():
    """验证BiFPN模型"""
    print("\n" + "=" * 70)
    print("📊 验证 BiFPN 模型...")
    print("=" * 70)

    model_path = os.path.join('..', 'runs', 'train_bifpn', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f"❌ 未找到权重文件: {model_path}")
        return None

    try:
        model = YOLO(model_path)
        results = model.val(
            data=os.path.join('..', 'data_augmented', 'grape_augmented.yaml'),
            split='val'
        )

        print("\n📈 BiFPN模型性能:")
        print(f"   mAP@0.5: {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"   Precision: {results.box.mp:.4f}")
        print(f"   Recall: {results.box.mr:.4f}")

        return results

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return None


if __name__ == "__main__":
    # 训练
    train_results = train_baseline_bifpn()

    if train_results:
        # 验证
        val_results = validate_bifpn()

        if val_results:
            print("\n🎉 Baseline+BiFPN训练和验证完成!")
    else:
        print("\n❌ 训练失败")