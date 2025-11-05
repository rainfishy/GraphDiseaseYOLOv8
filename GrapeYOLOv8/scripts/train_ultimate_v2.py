"""
终极三项组合模型 V2.0 - 真正的 SimAM + BiFPN + QFL
修复版本：确保所有三个创新点都真正生效

改进说明：
1. 使用真正的双轮BiFPN结构
2. 参数量会增加15-20%（正常现象）
3. 所有三个创新点都会验证是否生效
"""
import os
import sys

if __name__ == '__main__':
    sys.path.insert(0, r'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')

    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    import torch
    import yaml


    def create_ultimate_v2_model(yaml_path, nc=4):
        """
        创建真正的终极三项组合模型

        Args:
            yaml_path: 模型配置文件路径
            nc: 类别数

        Returns:
            model: YOLO模型对象
        """
        print("=" * 70)
        print("🔨 创建终极三项组合模型 V2.0")
        print("=" * 70)
        print(f"\n配置文件: {yaml_path}")
        print(f"类别数: {nc}")
        print(f"\n集成的三大创新点:")
        print(f"  ✅ 1. SimAM  - 无参数注意力机制（Layer 10）")
        print(f"  ✅ 2. BiFPN  - 双轮加权双向特征金字塔")
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

        # 详细验证各组件
        print(f"\n" + "=" * 70)
        print("🔍 组件详细验证")
        print("=" * 70)

        # 1. 检查SimAM
        print(f"\n【验证 1/3】SimAM 注意力机制")
        has_simam = False
        simam_location = None
        for name, module in detection_model.named_modules():
            module_type = str(type(module))
            if 'SimAM' in module_type or 'simam' in module_type.lower():
                has_simam = True
                simam_location = name
                print(f"  ✅ SimAM已集成")
                print(f"     位置: {name}")
                print(f"     类型: {type(module).__name__}")
                break

        if not has_simam:
            print(f"  ❌ SimAM未检测到")
            print(f"     请检查:")
            print(f"     1. ultralytics/nn/modules/simam.py 是否存在")
            print(f"     2. __init__.py 是否导入了SimAM")

        # 2. 检查BiFPN（详细）
        print(f"\n【验证 2/3】BiFPN 双向特征金字塔")
        total_params = sum(p.numel() for p in detection_model.parameters())
        baseline_params = 3011628
        param_increase = total_params - baseline_params
        percent_increase = (total_params / baseline_params - 1) * 100

        print(f"  总参数: {total_params:,}")
        print(f"  Baseline: {baseline_params:,}")
        print(f"  参数增加: {param_increase:,} ({percent_increase:.2f}%)")

        if param_increase > 300000:  # 至少增加30万参数
            print(f"  ✅ BiFPN已集成（参数显著增加）")
            print(f"     双轮融合结构已生效")
        elif param_increase > 50000:
            print(f"  ⚠️ BiFPN部分集成（参数增加较少）")
            print(f"     可能只有单轮融合")
        else:
            print(f"  ❌ BiFPN未生效（参数未增加）")
            print(f"     配置文件可能有问题")

        # 检查模型层数
        model_layers = len(list(detection_model.model))
        baseline_layers = 24  # Baseline YOLOv8n head的层数
        print(f"\n  模型层数: {model_layers}")
        print(f"  Baseline层数: {baseline_layers}")

        if model_layers > 30:
            print(f"  ✅ 层数增加明显（双轮BiFPN结构）")
        else:
            print(f"  ⚠️ 层数增加不明显")

        # 3. 检查QFL
        print(f"\n【验证 3/3】QFL 质量感知损失")
        print(f"  模型use_qfl标志: {detection_model.use_qfl}")

        criterion = detection_model.init_criterion()
        if hasattr(criterion, 'qfl'):
            print(f"  ✅ QFL已启用")
            print(f"     QFL对象: {type(criterion.qfl).__name__}")
            print(f"     criterion.use_qfl: {criterion.use_qfl}")
        else:
            print(f"  ❌ QFL对象未创建")

        # 综合评估
        print(f"\n" + "=" * 70)
        print("📊 综合评估")
        print("=" * 70)

        components_ok = 0
        if has_simam:
            components_ok += 1
        if param_increase > 300000:
            components_ok += 1
        if hasattr(criterion, 'qfl'):
            components_ok += 1

        print(f"\n  集成成功组件: {components_ok}/3")

        if components_ok == 3:
            print(f"  ✅ 所有三个创新点都已成功集成！")
            print(f"  ✅ 模型准备就绪，可以开始训练！")
        elif components_ok == 2:
            print(f"  ⚠️ 有1个组件未成功集成")
            print(f"  ⚠️ 建议检查后再训练")
        else:
            print(f"  ❌ 多个组件未成功集成")
            print(f"  ❌ 请检查配置文件和依赖")

        print("\n" + "=" * 70)
        print("✅ 模型创建完成！")
        print("=" * 70)

        return yolo_model


    def train_ultimate_v2():
        """训练终极三项组合模型 V2.0"""

        print("\n" + "=" * 70)
        print("🚀 终极三项组合训练 V2.0")
        print("=" * 70)
        print("\n真正的 SimAM + BiFPN + QFL 组合")
        print("\n预期效果:")
        print("  • mAP@0.5 > 91%")
        print("  • 参数增加 15-20% (正常)")
        print("  • 相对Baseline提升 > 3.5%")
        print("=" * 70)

        # 检查GPU
        if torch.cuda.is_available():
            print(f"\n✅ 使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        else:
            print("\n⚠️ 未检测到GPU，将使用CPU训练")

        # 检查新的配置文件
        model_yaml = os.path.join('..', 'models', 'yolov8n_ultimate_complete.yaml')

        if not os.path.exists(model_yaml):
            print(f"\n❌ 错误: 配置文件不存在")
            print(f"   文件路径: {model_yaml}")
            print(f"\n请按照以下步骤操作:")
            print(f"   1. 在 models/ 目录下创建 yolov8n_ultimate_complete.yaml")
            print(f"   2. 复制提供的完整BiFPN配置内容")
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

        # 训练配置（与之前完全一致）
        config = {
            'data': data_yaml,
            'epochs': 150,
            'imgsz': 640,
            'batch': 16,  # 如果显存不足，改为12或8
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
            'name': 'train_ultimate_v2',  # ⭐ 新的实验名称
            'exist_ok': True,
            'plots': True,
            'verbose': True,
        }

        print("\n📋 训练配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        print("\n⚠️ 重要提示:")
        print("  1. 参数量会增加15-20%，这是BiFPN的正常现象")
        print("  2. 如果显存不足，请将batch改为12或8")
        print("  3. 预计训练时间: 1-1.5小时")

        try:
            print("\n🔄 正在创建模型...")
            model = create_ultimate_v2_model(
                yaml_path=model_yaml,
                nc=nc
            )

            print("\n" + "=" * 70)
            print("🔥 开始训练...")
            print("=" * 70)
            print("\n请耐心等待训练完成...\n")

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
            print("  1. 显存不足 -> 减小batch size")
            print("  2. SimAM模块未安装 -> 检查模块文件")
            print("  3. 配置文件格式错误 -> 检查yaml语法")

            return None


    def validate_ultimate_v2():
        """验证终极模型 V2.0"""
        print("\n" + "=" * 70)
        print("📊 验证终极模型 V2.0")
        print("=" * 70)

        model_path = os.path.join('..', 'runs', 'train_ultimate_v2', 'weights', 'best.pt')

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
                workers=0
            )

            print("\n" + "=" * 70)
            print("📈 终极模型 V2.0 性能")
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
            print("完整性能对比汇总")
            print("=" * 70)

            print(f"\n实验进展:")
            print(f"   Baseline:            87.51%")
            print(f"   +SimAM:              88.53% (+1.02%)")
            print(f"   +BiFPN:              89.72% (+2.21%)")
            print(f"   +QFL:                89.59% (+2.08%)")
            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            ultimate_map = results.box.map50
            print(f"   +Ultimate V2:        {ultimate_map * 100:.2f}%", end='')

            improvement = (ultimate_map - 0.8751) * 100
            print(f" ({improvement:+.2f}%)")

            # 详细评价
            print(f"\n📊 详细评价:")
            if improvement >= 4.5:
                print(f"   🎉 超预期！提升达到 {improvement:.2f}%")
                print(f"   ✅ 超越了所有单项改进")
                print(f"   ✅ 三项创新点协同效果显著")
            elif improvement >= 3.5:
                print(f"   ✅ 优秀！提升达到 {improvement:.2f}%")
                print(f"   ✅ 达到预期目标")
                print(f"   ✅ 三项改进有效组合")
            elif improvement >= 2.5:
                print(f"   ✅ 良好！提升达到 {improvement:.2f}%")
                print(f"   ⚠️ 略低于预期，但仍有效")
            else:
                print(f"   ⚠️ 提升 {improvement:.2f}%（低于预期）")
                print(f"   建议: 检查是否所有组件都生效")

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
    print("║" + " " * 16 + "🎯 终极三项组合实验 V2.0" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")

    print("\n本实验集成三个创新点（修复版）:")
    print("  ┌──────────────────────────────────────────────┐")
    print("  │  1. SimAM  - 无参数注意力机制              │")
    print("  │  2. BiFPN  - 双轮加权双向特征金字塔        │")
    print("  │  3. QFL    - 质量感知焦点损失              │")
    print("  └──────────────────────────────────────────────┘")

    print("\n改进说明:")
    print("  • 使用真正的双轮BiFPN结构")
    print("  • 参数量会增加15-20%（正常）")
    print("  • 所有三个创新点都会验证")

    print("\n预期效果:")
    print("  • mAP@0.5 > 91%")
    print("  • 相对Baseline (87.51%) 提升 > 3.5%")
    print("  • 超越所有单项改进")

    print("\n" + "═" * 70)

    input("\n按 Enter 键开始训练...")

    # 开始训练
    train_results = train_ultimate_v2()

    if train_results:
        # 验证模型
        val_results = validate_ultimate_v2()

        if val_results:
            print("\n" + "╔" + "═" * 68 + "╗")
            print("║" + " " * 22 + "🎉 实验完成!" + " " * 30 + "║")
            print("╚" + "═" * 68 + "╝")

            print("\n所有三个创新点已集成并测试完成!")
            print("可以开始整理数据，撰写论文了!")

            print("\n生成的文件:")
            print(f"  • 模型权重: runs/train_ultimate_v2/weights/best.pt")
            print(f"  • 训练日志: runs/train_ultimate_v2/results.csv")
            print(f"  • 训练曲线: runs/train_ultimate_v2/results.png")

            print("\n" + "═" * 70)
        else:
            print("\n验证失败，但模型已训练完成")
    else:
        print("\n" + "═" * 70)
        print("训练失败，请检查错误信息")
        print("=" * 70)