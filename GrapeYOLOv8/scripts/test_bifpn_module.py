"""
测试BiFPN模块是否正确注册
"""
import sys

sys.path.insert(0, r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')


def test_bifpn_import():
    """测试BiFPN模块导入"""
    print("\n" + "=" * 70)
    print("🧪 测试BiFPN模块")
    print("=" * 70)

    try:
        # 测试导入
        print("\n[1/3] 测试模块导入...")
        from ultralytics.nn.modules.bifpn import WeightedFeatureFusion, BiFPNLayer
        print("  ✅ BiFPN模块导入成功")

        # 测试实例化
        print("\n[2/3] 测试模块实例化...")
        import torch

        fusion = WeightedFeatureFusion(num_inputs=2)
        print("  ✅ WeightedFeatureFusion 创建成功")

        bifpn = BiFPNLayer(in_channels=64, out_channels=128)
        print("  ✅ BiFPNLayer 创建成功")

        # 测试前向传播
        print("\n[3/3] 测试前向传播...")
        x1 = torch.randn(1, 64, 80, 80)
        x2 = torch.randn(1, 64, 80, 80)

        out_fusion = fusion(x1, x2)
        print(f"  ✅ WeightedFeatureFusion: {x1.shape} -> {out_fusion.shape}")

        x1 = torch.randn(1, 64, 80, 80)
        x2 = torch.randn(1, 64, 40, 40)
        out_bifpn = bifpn(x1, x2)
        print(f"  ✅ BiFPNLayer: {x1.shape} + {x2.shape} -> {out_bifpn.shape}")

        print("\n" + "=" * 70)
        print("🎉 所有测试通过！BiFPN模块工作正常！")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_bifpn_import()
    if not success:
        print("\n⚠️ 请检查上面的错误信息，修复后再继续")