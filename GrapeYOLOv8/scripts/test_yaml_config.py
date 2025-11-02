import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, 'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')

from ultralytics.nn.tasks import parse_model
import yaml
import torch


def test_yaml():
    print("\n" + "=" * 70)
    print("🔍 测试 yolov8n_simam.yaml 配置")
    print("=" * 70)

    # 使用绝对路径，确保能找到文件
    base_dir = 'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8'
    yaml_path = os.path.join(base_dir, 'GrapeYOLOv8/models/yolov8n_simam.yaml')

    # 检查文件是否存在
    if not os.path.exists(yaml_path):
        print(f"❌ 文件不存在: {yaml_path}")
        print("\n请检查以下位置:")
        print(f"  1. {os.path.join(base_dir, 'GrapeYOLOv8/models/')}")
        print(f"  2. {os.path.join(base_dir, 'models/')}")
        return

    print(f"✅ 找到配置文件: {yaml_path}")

    # 加载YAML
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print(f"\n✅ YAML配置加载成功")
    print(f"   - 模型规模: {cfg.get('scale', 'n')}")
    print(f"   - 类别数量: {cfg.get('nc', 80)}")

    # 解析模型
    try:
        model, save = parse_model(cfg, ch=[3])
        print(f"\n✅ 模型解析成功!")
        print(f"   - 总层数: {len(model)}")

        # 统计SimAM层
        simam_layers = [m for m in model.modules() if 'SimAM' in str(type(m))]
        print(f"   - SimAM层数: {len(simam_layers)}")

        # 测试前向传播
        print("\n[测试] 完整模型前向传播...")
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(x)
        print(f"✅ 前向传播成功!")
        print(f"   - 输入shape: {x.shape}")
        print(f"   - 输出类型: {type(output)}")

        print("\n" + "=" * 70)
        print("🎉 yolov8n_simam.yaml 配置验证通过!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 模型解析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_yaml()

