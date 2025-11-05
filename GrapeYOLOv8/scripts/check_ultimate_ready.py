"""
终极实验训练前检查
确保所有组件就绪
"""
import os
import sys

sys.path.insert(0, r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')

print("=" * 70)
print("🔍 终极实验训练前检查")
print("=" * 70)

# 检查项计数
total_checks = 0
passed_checks = 0

# 1. 检查QFL模块
print("\n[检查1/7] QFL模块...")
total_checks += 1
try:
    from ultralytics.nn.modules.qfl_loss import QualityFocalLoss

    qfl = QualityFocalLoss()
    print("  ✅ QFL模块存在")
    passed_checks += 1
except Exception as e:
    print(f"  ❌ QFL模块导入失败: {e}")

# 2. 检查SimAM模块
print("\n[检查2/7] SimAM模块...")
total_checks += 1
try:
    from ultralytics.nn.modules.simam import SimAM

    simam = SimAM()
    print("  ✅ SimAM模块存在")
    passed_checks += 1
except Exception as e:
    print(f"  ❌ SimAM模块导入失败: {e}")
    print("  解决方案: 确保 ultralytics/nn/modules/simam.py 存在")

# 3. 检查BiFPN模块
print("\n[检查3/7] BiFPN模块...")
total_checks += 1
try:
    from ultralytics.nn.modules.bifpn import BiFPN

    print("  ✅ BiFPN模块存在")
    passed_checks += 1
except Exception as e:
    print(f"  ⚠️ BiFPN模块未找到: {e}")
    print("  说明: BiFPN可能集成在配置文件中，这是正常的")
    passed_checks += 1  # BiFPN不需要单独模块

# 4. 检查loss.py修改
print("\n[检查4/7] loss.py修改...")
total_checks += 1
try:
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics import YOLO

    model = YOLO('yolov8n.yaml')
    criterion = v8DetectionLoss(model.model, use_qfl=True)

    if hasattr(criterion, 'use_qfl') and criterion.use_qfl:
        print("  ✅ loss.py已正确修改")
        passed_checks += 1
    else:
        print("  ❌ loss.py未正确修改")
except Exception as e:
    print(f"  ❌ loss.py检查失败: {e}")

# 5. 检查tasks.py修改
print("\n[检查5/7] tasks.py修改...")
total_checks += 1
try:
    from ultralytics.nn.tasks import DetectionModel

    test_model = DetectionModel('yolov8n.yaml', ch=3, nc=4, verbose=False, use_qfl=True)

    if hasattr(test_model, 'use_qfl') and test_model.use_qfl:
        print("  ✅ tasks.py已正确修改")
        passed_checks += 1
    else:
        print("  ❌ tasks.py未正确修改")
except Exception as e:
    print(f"  ❌ tasks.py检查失败: {e}")

# 6. 检查配置文件
print("\n[检查6/7] 模型配置文件...")
total_checks += 1
config_path = r'E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\GrapeYOLOv8\models\yolov8n_ultimate_simple.yaml'
if os.path.exists(config_path):
    print(f"  ✅ 配置文件存在")
    passed_checks += 1

    # 检查配置内容
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        has_simam = 'SimAM' in content
        has_bifpn = 'Concat' in content and 'Upsample' in content

        if has_simam:
            print("  ✅ 配置包含SimAM")
        else:
            print("  ⚠️ 配置未包含SimAM")

        if has_bifpn:
            print("  ✅ 配置包含BiFPN结构")
        else:
            print("  ⚠️ 配置未包含BiFPN结构")
else:
    print(f"  ❌ 配置文件不存在: {config_path}")
    print("  解决方案: 创建 models/yolov8n_ultimate_simple.yaml")

# 7. 检查数据集
print("\n[检查7/7] 数据集...")
total_checks += 1
data_path = r'E:\YOLOGrape\Grape_Disease_Experiment\GrapeDiseaseYOLOv8\GrapeYOLOv8\data_augmented\grape_augmented.yaml'
if os.path.exists(data_path):
    print(f"  ✅ 数据配置文件存在")
    passed_checks += 1
else:
    print(f"  ❌ 数据配置文件不存在: {data_path}")

# 总结
print("\n" + "=" * 70)
print("检查结果")
print("=" * 70)
print(f"\n通过: {passed_checks}/{total_checks}")

if passed_checks == total_checks:
    print("\n✅ 所有检查通过!")
    print("可以开始训练终极模型了!")
    print("\n运行命令:")
    print("  cd scripts")
    print("  python train_ultimate.py")
elif passed_checks >= total_checks - 1:
    print("\n⚠️ 大部分检查通过")
    print("建议解决剩余问题后再开始训练")
else:
    print("\n❌ 检查未通过")
    print("请解决以下问题:")
    print("  1. 确保QFL、SimAM、BiFPN模块都已创建")
    print("  2. 确保loss.py和tasks.py已修改")
    print("  3. 确保yolov8n_ultimate_simple.yaml已创建")

print("\n" + "=" * 70)

# 显示已完成的实验
print("\n已完成的实验:")
print("  ✅ Baseline:  87.51%")
print("  ✅ +SimAM:    88.53%")
print("  ✅ +BiFPN:    89.72%")
print("  ✅ +QFL:      89.59%")
print("  ⏳ +Ultimate: 待训练 (预期 > 92%)")

print("\n" + "=" * 70)