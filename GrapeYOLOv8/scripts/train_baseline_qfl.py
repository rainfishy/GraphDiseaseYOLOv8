"""
训练 Baseline + QFL (Quality Focal Loss) 模型
修复版：正确启用QFL
"""
import os
import sys

sys.path.insert(0, r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')


from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import torch
import yaml


def create_qfl_model(yaml_path='yolov8n.yaml', nc=4, use_qfl=True):
    """
    创建启用QFL的模型

    Args:
        yaml_path: 模型配置文件路径
        nc: 类别数量
        use_qfl: 是否使用QFL

    Returns:
        model: YOLO模型对象
    """
    # 方法1: 直接使用DetectionModel (推荐)
    print(f"🔨 创建模型: {yaml_path}")
    print(f"   类别数: {nc}")
    print(f"   使用QFL: {use_qfl}")

    # 创建DetectionModel，传入use_qfl参数
    detection_model = DetectionModel(
        cfg=yaml_path,
        ch=3,
        nc=nc,
        verbose=True,
        use_qfl=use_qfl  # ⭐ 关键：在初始化时传入
    )

    # 创建YOLO对象并替换model
    yolo_model = YOLO(yaml_path)
    yolo_model.model = detection_model

    # 验证QFL是否启用
    print(f"\n✅ 模型创建成功!")
    print(f"   model.use_qfl = {detection_model.use_qfl}")

    # 初始化criterion来验证QFL
    criterion = detection_model.init_criterion()
    print(f"   criterion.use_qfl = {criterion.use_qfl}")
    if hasattr(criterion, 'qfl'):
        print(f"   ✅ QFL对象已创建: {type(criterion.qfl).__name__}")

    return yolo_model


def train_baseline_qfl():
    """训练YOLOv8n+QFL"""

    print("=" * 70)
    print("🚀 训练 YOLOv8n + QFL 模型")
    print("=" * 70)

    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("\n⚠️ 未检测到GPU")

    # 读取数据配置获取类别数
    data_yaml = os.path.join('..', 'data_augmented', 'grape_augmented.yaml')
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    nc = data_config.get('nc', 4)
    print(f"\n📊 数据集配置:")
    print(f"   类别数: {nc}")
    print(f"   类别名: {data_config.get('names', [])}")

    # 训练配置
    config = {
        'data': data_yaml,
        'epochs': 150,
        'imgsz': 640,
        'batch': 16,
        'workers': 4,
        'patience': 20,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # 损失函数权重（保持默认）
        'cls': 0.5,  # 分类损失权重
        'box': 7.5,  # 边界框损失权重
        'dfl': 1.5,  # DFL损失权重

        'save': True,
        'save_period': 10,
        'project': os.path.join('..', 'runs'),
        'name': 'train_qfl',
        'exist_ok': True,
        'plots': True,
        'verbose': True,
    }

    print("\n📋 训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    try:
        print("\n🔄 创建QFL模型...")

        # ⭐ 关键：使用create_qfl_model创建模型
        model = create_qfl_model(
            yaml_path='yolov8n.yaml',
            nc=nc,
            use_qfl=True  # 启用QFL
        )

        print("\n" + "=" * 70)
        print("🔥 开始训练...")
        print("=" * 70 + "\n")

        results = model.train(**config)

        print("\n" + "=" * 70)
        print("✅ QFL模型训练完成!")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_qfl():
    """验证QFL模型"""
    print("\n" + "=" * 70)
    print("📊 验证 QFL 模型...")
    print("=" * 70)

    model_path = os.path.join('..', 'runs', 'train_qfl', 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f"❌ 未找到权重文件: {model_path}")
        return None

    try:
        model = YOLO(model_path)
        results = model.val(
            data=os.path.join('..', 'data_augmented', 'grape_augmented.yaml'),
            split='val'
        )

        print("\n📈 QFL模型性能:")
        print(f"   mAP@0.5: {results.box.map50:.4f}")
        print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"   Precision: {results.box.mp:.4f}")
        print(f"   Recall: {results.box.mr:.4f}")

        # 各类别性能
        print("\n📊 各类别性能:")
        class_names = ['black_rot', 'blight', 'black_measles', 'Healthy']
        for i, name in enumerate(class_names):
            if i < len(results.box.ap_class_index):
                ap = results.box.ap50[i]
                print(f"   {name}: {ap:.4f}")

        return results

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("📌 重要提示:")
    print("=" * 70)
    print("训练前请确保已完成以下步骤:")
    print("1. ✅ 创建 ultralytics/nn/modules/qfl_loss.py")
    print("2. ✅ 修改 ultralytics/utils/loss.py (3处修改)")
    print("3. ✅ 修改 ultralytics/nn/tasks.py (3处修改)")
    print("4. ✅ 运行 test_qfl_complete.py 验证通过")
    print("=" * 70)

    input("\n按Enter键开始训练...")

    # 训练
    train_results = train_baseline_qfl()

    if train_results:
        # 验证
        val_results = validate_qfl()

        if val_results:
            print("\n" + "=" * 70)
            print("🎉 Baseline+QFL训练和验证完成!")
            print("=" * 70)
            print("\n📊 性能对比:")
            print(f"   Baseline:  87.51%")
            print(f"   +SimAM:    88.53%")
            print(f"   +BiFPN:    89.72%")
            print(f"   +QFL:      {val_results.box.map50:.2%}")

            # 计算提升
            baseline_map = 0.8751
            qfl_map = val_results.box.map50
            improvement = (qfl_map - baseline_map) * 100
            print(f"\n📈 相对Baseline提升: {improvement:+.2f}%")
    else:
        print("\n❌ 训练失败")