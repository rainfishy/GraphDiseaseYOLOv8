import subprocess
import sys


def setup_yolov8_environment():
    """安装YOLOv8训练环境"""

    packages = [
        "ultralytics==8.2.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "thop",  # 计算FLOPs
        "seaborn",  # 可视化
        "pandas"
    ]

    print("=" * 70)
    print("🚀 YOLOv8环境配置")
    print("=" * 70)

    # 检查CUDA
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch未安装")

    # 安装包
    for package in packages:
        try:
            if "==" in package:
                pkg_name = package.split("==")[0]
            else:
                pkg_name = package

            __import__(pkg_name)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"📦 安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # 验证Ultralytics安装
    try:
        from ultralytics import YOLO
        print("🎉 YOLOv8环境配置成功!")

        # 测试模型加载
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8n模型加载测试通过")

    except Exception as e:
        print(f"❌ 环境验证失败: {e}")


if __name__ == "__main__":
    setup_yolov8_environment()