"""
测试YOLOv8n+BiFPN的yaml配置
"""
import sys
import os

sys.path.insert(0, r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')

from ultralytics import YOLO
import torch


def test_bifpn_yaml():
    """测试BiFPN模型配置"""
    print("\n" + "=" * 70)
    print("🔍 测试 yolov8n_bifpn.yaml 配置")
    print("=" * 70)

    yaml_path = r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8/GrapeYOLOv8/models/yolov8n_bifpn.yaml'

    if not os.path.exists(yaml_path):
        print(f"❌ 文件不存在: {yaml_path}")
        return False

    print(f"\n✅ 找到配置文件: {yaml_path}")

    try:
        # 加载模型
        print("\n[1/2] 加载模型...")
        model = YOLO(yaml_path)
        print("  ✅ 模型加载成功!")

        # 测试前向传播
        print("\n[2/2] 测试前向传播...")
        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            results = model.predict(x, verbose=False)

        print("  ✅ 前向传播成功!")
        print(f"  - 输入shape: {x.shape}")
        print(f"  - 检测结果: {len(results)}")

        print("\n" + "=" * 70)
        print("🎉 yolov8n_bifpn.yaml 配置验证通过!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_bifpn_yaml()
    if not success:
        print("\n⚠️ 配置有问题，请检查错误信息")
    else:
        print("\n✅ 可以开始训练了！")