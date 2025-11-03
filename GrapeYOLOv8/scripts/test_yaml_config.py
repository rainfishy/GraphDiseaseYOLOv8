import sys
import os

sys.path.insert(0, r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')

from ultralytics import YOLO
import torch


def test_yaml_config():
    """测试yolov8n_simam.yaml配置"""
    print("\n" + "=" * 70)
    print("🔍 测试 yolov8n_simam.yaml 配置")
    print("=" * 70)

    # 使用绝对路径
    base_dir = 'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8'
    yaml_path = os.path.join(base_dir, 'GrapeYOLOv8/models/yolov8n_simam.yaml')

    print(f"\n✅ 加载配置: {yaml_path}")

    # 使用YOLO类加载模型（这个会正确处理Concat）
    try:
        model = YOLO(yaml_path)

        print(f"\n✅ 模型加载成功!")

        # 测试前向传播
        print("\n[测试] 完整模型前向传播...")
        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            results = model.predict(x, verbose=False)

        print(f"✅ 前向传播成功!")
        print(f"   - 输入shape: {x.shape}")
        print(f"   - 检测结果数量: {len(results)}")

    except Exception as e:
        print(f"\n❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("🎉 yolov8n_simam.yaml 配置验证通过!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_yaml_config()