import os
from ultralytics import YOLO
import torch

def train_baseline_model():
    print("=" * 70)
    print("🎯 开始训练YOLOv8n基线模型")
    print("=" * 70)
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
        'name': 'baseline_yolov8n',
        'exist_ok': True
    }
    print("📋 训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    try:
        print("🔄 加载YOLOv8n预训练模型...")
        model = YOLO('yolov8n.pt')
        print("🔥 开始训练...")
        results = model.train(**config)
        print("=" * 70)
        print("✅ 基线模型训练完成!")
        print("=" * 70)
        return results
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")

def validate_baseline():
    print("📊 验证基线模型...")
    model_path = os.path.join('..', 'runs', 'baseline_yolov8n', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        print(f"❌ 未找到权重文件: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        results = model.val(
            data=os.path.join('..', 'data_augmented', 'grape_augmented.yaml'),
            split='val'
        )
        print("📈 基线模型性能:")
        print(f"   mAP@0.5: {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"   精确率: {results.box.p:.4f}")
        print(f"   召回率: {results.box.r:.4f}")
        return results
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")

if __name__ == "__main__":
    train_results = train_baseline_model()
    val_results = validate_baseline()