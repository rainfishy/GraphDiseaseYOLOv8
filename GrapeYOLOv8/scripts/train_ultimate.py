"""
终极全组合实验：Baseline + SimAM + BiFPN + QFL
集成三个最优改进点

使用方法:
1. 确保 models/yolov8n_ultimate_simple.yaml 已创建
2. 运行此脚本: python train_ultimate.py
"""
import os
import sys

if __name__ == '__main__':
    sys.path.insert(0, r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')

    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    import torch
    import yaml


    def create_ultimate_model(yaml_path, nc=4):
        """
        创建终极全组合模型：SimAM + BiFPN + QFL

        Args:
            yaml_path: 模型配置文件路径
            nc: 类别数

        Returns:
            model: YOLO模型对象
        """
        print("=" * 70)
        print("🔨 创建终极全组合模型")
        print("=" * 70)
        print(f"\n配置文件: {yaml_path}")
        print(f"类别数: {nc}")
        print(f"\n集成的创新点:")
        print(f"  ✅ 1. SimAM  - 参数高效的注意力机制")
        print(f"  ✅ 2. BiFPN  - 加权双向特征金字塔")
        print(f"  ✅ 3. QFL    - 质量感知焦点损失")

        # 创建DetectionModel，启用QFL
        detection_model = DetectionModel(
            cfg=yaml_path,
            ch=3,
            nc=nc,
            verbose=True,
            use_qfl=True  # ⭐ 启用QFL损失
        )

        # 创建YOLO对象
        yolo_model = YOLO(yaml_path)
        yolo_model.model = detection_model

        # 验证各组件
        print(f"\n" + "=" * 70)
        print("组件验证")
        print("=" * 70)

        # 检查SimAM
        has_simam = False
        for name, module in detection_model.named_modules():
            module_type = str(type(module))
            if 'SimAM' in module_type or 'simam' in module_type.lower():
                has_simam = True
                print(f"✅ SimAM: 已集成 (位置: {name})")
                break
        if not has_simam:
            print(f"⚠️ SimAM: 未检测到")
            print(f"   请检查:")
            print(f"   1. ultralytics/nn/modules/simam.py 是否存在")
            print(f"   2. __init__.py 是否导入了SimAM")
            print(f"   3. yaml配置中是否包含SimAM层")

        # 检查BiFPN（通过层数判断）
        baseline_layers = 129  # Baseline YOLOv8n的层数
        model_layers = len(list(detection_model.model))
        if model_layers > baseline_layers:
            print(f"✅ BiFPN: 已集成 (模型层数: {model_layers}, Baseline: {baseline_layers})")
        else:
            print(f"⚠️ BiFPN: 未检测到明显增加")
            print(f"   模型层数: {model_layers} (Baseline: {baseline_layers})")

        # 检查QFL
        print(f"✅ QFL: {detection_model.use_qfl}")
        criterion = detection_model.init_criterion()
        if hasattr(criterion, 'qfl'):
            print(f"   QFL对象: {type(criterion.qfl).__name__}")
            print(f"   criterion.use_qfl: {criterion.use_qfl}")
        else:
            print(f"   ⚠️ QFL对象未创建")

        # 显示模型统计
        total_params = sum(p.numel() for p in detection_model.parameters())
        baseline_params = 3011628
        print(f"\n📊 模型统计:")
        print(f"   总参数: {total_params:,}")
        print(f"   Baseline: {baseline_params:,}")
        print(f"   增加: {(total_params - baseline_params):,} ({((total_params / baseline_params - 1) * 100):.1f}%)")

        print("\n" + "=" * 70)
        print("✅ 终极模型创建成功!")
        print("=" * 70)

        return yolo_model


    def train_ultimate():
        """训练终极全组合模型"""

        print("\n" + "=" * 70)
        print("🚀 终极全组合实验")
        print("=" * 70)
        print("\nSimAM + BiFPN + QFL")
        print("\n预期效果: mAP@0.5 > 92%")
        print("预期提升: +4.5% 以上 (相对Baseline 87.51%)")
        print("=" * 70)

        # 检查GPU
        if torch.cuda.is_available():
            print(f"\n✅ 使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        else:
            print("\n⚠️ 未检测到GPU，将使用CPU训练")

        # 检查模型配置文件
        model_yaml = os.path.join('..', 'models', 'yolov8n_ultimate_simple.yaml')
        if not os.path.exists(model_yaml):
            print(f"\n❌ 错误: 模型配置文件不存在")
            print(f"   文件路径: {model_yaml}")
            print(f"\n请按照以下步骤操作:")
            print(f"   1. 在 models/ 目录下创建 yolov8n_ultimate_simple.yaml")
            print(f"   2. 复制提供的配置内容到该文件")
            print(f"   3. 重新运行此脚本")
            return None

        print(f"\n✅ 配置文件: {model_yaml}")

        # 读取数据配置
        data_yaml = os.path.join('..', 'data_augmented', 'grape_augmented.yaml')
        if not os.path.exists(data_yaml):
            print(f"\n❌ 数据配置文件不存在: {data_yaml}")
            return None

        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        nc = data_config.get('nc', 4)

        print(f"\n📊 数据集配置:")
        print(f"   路径: {data_yaml}")
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

            # 学习率设置
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,

            # 损失权重
            'cls': 0.5,
            'box': 7.5,
            'dfl': 1.5,

            # 保存设置
            'save': True,
            'save_period': 10,
            'project': os.path.join('..', 'runs'),
            'name': 'train_ultimate',
            'exist_ok': True,
            'plots': True,
            'verbose': True,
        }

        print("\n📋 训练配置:")
        for key, value in config.items():
            if key in ['data', 'project']:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")

        try:
            print("\n🔄 正在创建模型...")
            model = create_ultimate_model(
                yaml_path=model_yaml,
                nc=nc
            )

            print("\n" + "=" * 70)
            print("🔥 开始训练...")
            print("=" * 70)
            print("\n预计训练时间: 约1-1.5小时")
            print("请耐心等待训练完成...\n")

            results = model.train(**config)

            print("\n" + "=" * 70)
            print("✅ 训练完成!")
            print("=" * 70)

            return results

        except Exception as e:
            print(f"\n❌ 训练失败: {e}")
            print("\n错误详情:")
            import traceback
            traceback.print_exc()

            print("\n可能的原因:")
            print("  1. SimAM模块未正确安装")
            print("  2. BiFPN配置有误")
            print("  3. 显存不足")
            print("\n解决建议:")
            print("  1. 检查 ultralytics/nn/modules/simam.py")
            print("  2. 检查模型配置文件")
            print("  3. 减小batch size (config['batch'] = 8)")

            return None


    def validate_ultimate():
        """验证终极模型"""
        print("\n" + "=" * 70)
        print("📊 验证终极模型")
        print("=" * 70)

        model_path = os.path.join('..', 'runs', 'train_ultimate', 'weights', 'best.pt')

        if not os.path.exists(model_path):
            print(f"\n❌ 模型文件不存在: {model_path}")
            print("   请确保训练已完成")
            return None

        try:
            print(f"\n✅ 加载模型: {model_path}")
            model = YOLO(model_path)

            print("\n正在验证...")
            results = model.val(
                data=os.path.join('..', 'data_augmented', 'grape_augmented.yaml'),
                split='val',
                workers=0  # 避免Windows多进程问题
            )

            print("\n" + "=" * 70)
            print("📈 终极模型性能")
            print("=" * 70)

            print(f"\n🎯 整体指标:")
            print(f"   mAP@0.5:      {results.box.map50:.4f} ({results.box.map50 * 100:.2f}%)")
            print(f"   mAP@0.5:0.95: {results.box.map:.4f} ({results.box.map * 100:.2f}%)")
            print(f"   Precision:    {results.box.mp:.4f}")
            print(f"   Recall:       {results.box.mr:.4f}")

            print(f"\n📊 各类别 mAP@0.5:")
            class_names = ['black_rot', 'blight', 'black_measles', 'Healthy']
            for i, name in enumerate(class_names):
                if i < len(results.box.ap50):
                    ap = results.box.ap50[i]
                    print(f"   {name:15s}: {ap:.4f} ({ap * 100:.2f}%)")

            print("\n" + "=" * 70)
            print("性能对比汇总")
            print("=" * 70)

            print(f"\n实验进展:")
            print(f"   Baseline:        87.51%")
            print(f"   +SimAM:          88.53% (+1.02%)")
            print(f"   +BiFPN:          89.72% (+2.21%)")
            print(f"   +QFL:            89.59% (+2.08%)")
            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            ultimate_map = results.box.map50
            print(f"   +Ultimate:       {ultimate_map * 100:.2f}%", end='')

            improvement = (ultimate_map - 0.8751) * 100
            print(f" ({improvement:+.2f}%)")

            # 评价
            print(f"\n评价:")
            if improvement >= 4.5:
                print(f"   🎉 超预期! 提升达到 {improvement:.2f}%")
                print(f"   超越了所有单项改进!")
            elif improvement >= 3.5:
                print(f"   ✅ 优秀! 提升达到 {improvement:.2f}%")
            elif improvement >= 2.5:
                print(f"   ✅ 良好! 提升达到 {improvement:.2f}%")
            else:
                print(f"   ⚠️ 提升 {improvement:.2f}% (略低于预期)")
                print(f"   建议: 调整超参数或增加训练轮次")

            print("\n" + "=" * 70)
            print("✅ 验证完成!")
            print("=" * 70)

            return results

        except Exception as e:
            print(f"\n❌ 验证失败: {e}")
            import traceback
            traceback.print_exc()
            return None


    # ============ 主程序入口 ============
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "🎯 终极全组合实验" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")

    print("\n本实验将集成三个创新点:")
    print("  ┌─────────────────────────────────────────┐")
    print("  │  1. SimAM  - 参数高效的注意力机制      │")
    print("  │  2. BiFPN  - 加权双向特征金字塔        │")
    print("  │  3. QFL    - 质量感知焦点损失          │")
    print("  └─────────────────────────────────────────┘")

    print("\n预期效果:")
    print("  • mAP@0.5 > 92%")
    print("  • 相对Baseline (87.51%) 提升 > 4.5%")
    print("  • 超越所有单项改进")

    print("\n" + "═" * 70)

    input("\n按 Enter 键开始训练...")

    # 开始训练
    train_results = train_ultimate()

    if train_results:
        # 验证模型
        val_results = validate_ultimate()

        if val_results:
            print("\n" + "╔" + "═" * 68 + "╗")
            print("║" + " " * 22 + "🎉 实验完成!" + " " * 30 + "║")
            print("╚" + "═" * 68 + "╝")

            print("\n所有改进点已集成并测试完成!")
            print("可以开始整理数据，撰写论文了!")

            print("\n生成的文件:")
            print(f"  • 模型权重: runs/train_ultimate/weights/best.pt")
            print(f"  • 训练日志: runs/train_ultimate/results.csv")
            print(f"  • 训练曲线: runs/train_ultimate/results.png")

            print("\n" + "═" * 70)
        else:
            print("\n验证失败，但模型已训练完成")
            print("可以手动验证: python verify_ultimate.py")
    else:
        print("\n" + "═" * 70)
        print("训练失败，请检查错误信息")
        print("=" * 70)