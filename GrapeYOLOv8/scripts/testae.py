import sys

sys.path.insert(0, 'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')

from ultralytics.nn.tasks import parse_model
import yaml
import torch


def test_yaml():
    print("\n" + "=" * 70)
    print("🔍 测试 yolov8n_simam.yaml 配置")
    print("=" * 70)

    yaml_path = r'E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\GrapeYOLOv8\models\yolov8n_simam.yaml'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print(f"\n✅ YAML加载成功")
    print(f"   规模: {cfg.get('scale')}")
    print(f"   类别数: {cfg.get('nc')}")

    try:
        model, save = parse_model(cfg, ch=[3])
        print(f"\n✅ 模型解析成功!")
        print(f"   总层数: {len(model)}")

        # 统计SimAM层
        simam_count = sum(1 for m in model.modules() if 'SimAM' in str(type(m)))
        print(f"   SimAM层数: {simam_count}")

        # 测试前向传播
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(x)
        print(f"\n✅ 前向传播成功!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_yaml()