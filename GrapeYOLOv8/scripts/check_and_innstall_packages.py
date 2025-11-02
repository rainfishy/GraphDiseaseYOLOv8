import subprocess
import sys


def check_and_install_packages():
    """检查并安装必要的Python包"""

    required_packages = {
        'opencv-python': '4.8.1.78',
        'numpy': '1.24.3',
        'albumentations': '1.3.1',
        'tqdm': 'latest',
        'matplotlib': 'latest',
        'pyyaml': 'latest'
    }

    print("=" * 70)
    print("🔍 检查Python环境和依赖包")
    print("=" * 70)

    # 检查Python版本
    print(f"\n✅ Python版本: {sys.version.split()[0]}")

    # 检查每个包
    for package, version in required_packages.items():
        try:
            if package == 'opencv-python':
                import cv2
                installed_version = cv2.__version__
                package_name = 'opencv-python'
            elif package == 'numpy':
                import numpy as np
                installed_version = np.__version__
                package_name = 'numpy'
            elif package == 'albumentations':
                import albumentations as A
                installed_version = A.__version__
                package_name = 'albumentations'
            elif package == 'tqdm':
                import tqdm
                installed_version = tqdm.__version__
                package_name = 'tqdm'
            elif package == 'matplotlib':
                import matplotlib
                installed_version = matplotlib.__version__
                package_name = 'matplotlib'
            elif package == 'pyyaml':
                import yaml
                installed_version = yaml.__version__ if hasattr(yaml, '__version__') else 'installed'
                package_name = 'pyyaml'

            print(f"✅ {package_name}: {installed_version}")

        except ImportError:
            print(f"❌ {package} 未安装")
            print(f"   正在安装 {package}...")

            if version == 'latest':
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])

            print(f"✅ {package} 安装完成")

    print("\n" + "=" * 70)
    print("✅ 所有依赖包检查完成！")
    print("=" * 70)


if __name__ == "__main__":
    check_and_install_packages()