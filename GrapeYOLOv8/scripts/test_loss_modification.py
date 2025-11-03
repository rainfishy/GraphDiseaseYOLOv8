"""
测试 loss.py 修改是否正确
"""
import sys

sys.path.insert(0, r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')

print("=" * 70)
print("测试 loss.py 修改")
print("=" * 70)

# 测试1: 导入QFL
print("\n[测试1] 导入 QualityFocalLoss...")
try:
    from ultralytics.nn.modules.qfl_loss import QualityFocalLoss

    print("✅ QFL导入成功")
except Exception as e:
    print(f"❌ QFL导入失败: {e}")
    print("⚠️ 请先确保 qfl_loss.py 文件已创建")
    exit(1)

# 测试2: 导入loss模块
print("\n[测试2] 导入 loss 模块...")
try:
    from ultralytics.utils.loss import v8DetectionLoss

    print("✅ loss模块导入成功")
except Exception as e:
    print(f"❌ loss模块导入失败: {e}")
    print(f"   错误详情: {e}")
    exit(1)

# 测试3: 创建模型并测试QFL
print("\n[测试3] 测试 v8DetectionLoss 是否支持 use_qfl...")
try:
    from ultralytics import YOLO

    # 创建模型
    model = YOLO('yolov8n.yaml')

    # 测试创建loss（不使用QFL）
    print("   测试 use_qfl=False...")
    loss_no_qfl = v8DetectionLoss(model.model, use_qfl=False)
    print(f"   ✅ use_qfl=False: {loss_no_qfl.use_qfl}")

    # 测试创建loss（使用QFL）
    print("   测试 use_qfl=True...")
    loss_with_qfl = v8DetectionLoss(model.model, use_qfl=True)
    print(f"   ✅ use_qfl=True: {loss_with_qfl.use_qfl}")

    # 验证QFL对象是否创建
    if hasattr(loss_with_qfl, 'qfl'):
        print("   ✅ QFL对象已创建")
    else:
        print("   ❌ QFL对象未创建")
        exit(1)

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("🎉 所有测试通过！loss.py 修改成功！")
print("=" * 70)
print("\n✅ 下一步: 修改 tasks.py")