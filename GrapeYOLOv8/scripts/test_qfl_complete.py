"""
完整的QFL集成测试
测试loss.py和tasks.py的修改是否正确
"""
import sys

sys.path.insert(0, r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')


def test_qfl_integration():
    """完整的QFL集成测试"""

    print("=" * 70)
    print("🧪 完整QFL集成测试")
    print("=" * 70)

    # 测试1: QFL模块导入
    print("\n[测试1/6] 测试QFL模块导入...")
    try:
        from ultralytics.nn.modules.qfl_loss import QualityFocalLoss
        qfl = QualityFocalLoss(beta=2.0)
        print("✅ QFL模块导入成功")
    except Exception as e:
        print(f"❌ QFL模块导入失败: {e}")
        print("   请检查是否创建了 qfl_loss.py 文件")
        return False

    # 测试2: loss.py修改
    print("\n[测试2/6] 测试loss.py修改...")
    try:
        from ultralytics.utils.loss import v8DetectionLoss
        from ultralytics import YOLO

        # 创建临时模型
        model = YOLO('yolov8n.yaml')

        # 测试不带QFL
        criterion_no_qfl = v8DetectionLoss(model.model, use_qfl=False)
        assert hasattr(criterion_no_qfl, 'use_qfl'), "use_qfl属性不存在"
        assert criterion_no_qfl.use_qfl == False, "use_qfl应该为False"
        print("  ✅ 不带QFL的criterion创建成功")

        # 测试带QFL
        criterion_qfl = v8DetectionLoss(model.model, use_qfl=True)
        assert criterion_qfl.use_qfl == True, "use_qfl应该为True"
        assert hasattr(criterion_qfl, 'qfl'), "qfl属性不存在"
        print("  ✅ 带QFL的criterion创建成功")

        print("✅ loss.py修改验证通过")
    except Exception as e:
        print(f"❌ loss.py修改验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试3: tasks.py修改
    print("\n[测试3/6] 测试tasks.py修改...")
    try:
        from ultralytics.nn.tasks import DetectionModel

        # 测试不带QFL
        model_no_qfl = DetectionModel('yolov8n.yaml', ch=3, nc=4, verbose=False, use_qfl=False)
        assert hasattr(model_no_qfl, 'use_qfl'), "DetectionModel缺少use_qfl属性"
        assert model_no_qfl.use_qfl == False, "use_qfl应该为False"
        print("  ✅ 不带QFL的DetectionModel创建成功")

        # 测试带QFL
        model_qfl = DetectionModel('yolov8n.yaml', ch=3, nc=4, verbose=False, use_qfl=True)
        assert model_qfl.use_qfl == True, "use_qfl应该为True"
        print("  ✅ 带QFL的DetectionModel创建成功")

        print("✅ tasks.py修改验证通过")
    except Exception as e:
        print(f"❌ tasks.py修改验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试4: init_criterion传递参数
    print("\n[测试4/6] 测试init_criterion参数传递...")
    try:
        # 不带QFL
        model_no_qfl = DetectionModel('yolov8n.yaml', ch=3, nc=4, verbose=False, use_qfl=False)
        criterion_no_qfl = model_no_qfl.init_criterion()
        assert criterion_no_qfl.use_qfl == False, "criterion的use_qfl应该为False"
        print("  ✅ 不带QFL的参数传递正确")

        # 带QFL
        print("  创建带QFL的模型...")
        model_qfl = DetectionModel('yolov8n.yaml', ch=3, nc=4, verbose=False, use_qfl=True)
        criterion_qfl = model_qfl.init_criterion()
        assert criterion_qfl.use_qfl == True, "criterion的use_qfl应该为True"
        assert hasattr(criterion_qfl, 'qfl'), "criterion应该有qfl属性"
        print("  ✅ 带QFL的参数传递正确")

        print("✅ init_criterion参数传递验证通过")
    except Exception as e:
        print(f"❌ init_criterion参数传递验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试5: YOLO接口
    print("\n[测试5/6] 测试YOLO接口...")
    try:
        from ultralytics import YOLO

        model = YOLO('yolov8n.yaml')
        model.model.use_qfl = True

        criterion = model.model.init_criterion()
        assert criterion.use_qfl == True, "YOLO接口的QFL启用失败"

        print("✅ YOLO接口验证通过")
    except Exception as e:
        print(f"❌ YOLO接口验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试6: 前向传播
    print("\n[测试6/6] 测试前向传播和损失计算...")
    try:
        import torch
        from ultralytics import YOLO

        # 创建模型
        model = YOLO('yolov8n.yaml')
        model.model.use_qfl = True

        # 创建criterion
        criterion = model.model.init_criterion()

        # 创建模拟数据
        batch_size = 2
        x = torch.randn(batch_size, 3, 640, 640)

        # 模拟batch
        batch = {
            'batch_idx': torch.tensor([0, 0, 1], dtype=torch.long),
            'cls': torch.tensor([[0], [1], [2]], dtype=torch.float),
            'bboxes': torch.tensor([
                [0.5, 0.5, 0.2, 0.2],
                [0.3, 0.3, 0.1, 0.1],
                [0.7, 0.7, 0.15, 0.15]
            ], dtype=torch.float)
        }

        # 前向传播
        with torch.no_grad():
            preds = model.model(x)

        # 计算loss
        loss, loss_items = criterion(preds, batch)

        print(f"  ✅ 损失计算成功")
        # loss是一个包含3个元素的张量 [box_loss, cls_loss, dfl_loss]
        print(f"     Total loss: {loss.sum().item():.4f}")
        print(f"     Box loss: {loss[0].item():.4f}")
        print(f"     Cls loss: {loss[1].item():.4f}")
        print(f"     DFL loss: {loss[2].item():.4f}")
        print(f"     Loss items (detached): {loss_items}")
        print("✅ 前向传播和损失计算验证通过")

    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("\n🚀 开始完整QFL集成测试...\n")

    success = test_qfl_integration()

    if success:
        print("\n" + "=" * 70)
        print("🎉 所有测试通过! QFL集成成功!")
        print("=" * 70)
        print("\n✅ 现在可以运行以下命令开始训练:")
        print("   cd GrapeYOLOv8/scripts")
        print("   python train_baseline_qfl.py")
        print("\n预期训练时间: 约1-1.5小时")
        print("预期mAP@0.5提升: +0.8% (从87.51%到约88.3%)")
    else:
        print("\n" + "=" * 70)
        print("❌ 测试失败! 请检查以上错误信息")
        print("=" * 70)
        print("\n请确认:")
        print("1. ✅ qfl_loss.py 文件已创建")
        print("2. ✅ loss.py 已正确修改（3处）")
        print("3. ✅ tasks.py 已正确修改（3处）")
        print("\n如果需要帮助，请把错误信息发给我!")