"""
将SimAM注意力机制集成到YOLOv8中
"""

import os
import shutil
from pathlib import Path


def integrate_simam_to_yolov8():
    """将SimAM集成到Ultralytics YOLOv8中"""

    print("=" * 70)
    print("🔧 集成SimAM到YOLOv8")
    print("=" * 70)

    try:
        import ultralytics
        ultralytics_path = Path(ultralytics.__file__).parent
        print(f"✅ Ultralytics路径: {ultralytics_path}")
    except ImportError:
        print("❌ 未安装ultralytics")
        return

    # 1. 复制SimAM模块到nn/modules
    print("\n[步骤1] 复制SimAM模块...")

    src_file = Path(__file__).parent / "simam_module.py"
    dst_dir = ultralytics_path / "nn" / "modules"
    dst_file = dst_dir / "simam.py"

    if not src_file.exists():
        print(f"❌ 源文件不存在: {src_file}")
        return

    # 提取SimAM类定义
    simam_code = '''"""SimAM Attention Module for YOLOv8"""

import torch
import torch.nn as nn

class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module"""

    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (
            4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        return x * self.activation(y)
'''

    # 写入文件
    with open(dst_file, 'w', encoding='utf-8') as f:
        f.write(simam_code)

    print(f"✅ SimAM模块已复制到: {dst_file}")

    # 2. 修改__init__.py注册SimAM
    print("\n[步骤2] 注册SimAM模块...")

    init_file = dst_dir / "__init__.py"

    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已经添加
    if "SimAM" not in content:
        # 在导入部分添加
        import_line = "from .simam import SimAM"

        # 找到合适的位置插入
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('from .conv import'):
                lines.insert(i + 1, import_line)
                break

        # 在__all__中添加
        for i, line in enumerate(lines):
            if '__all__ =' in line:
                # 找到__all__列表
                j = i
                while j < len(lines) and ']' not in lines[j]:
                    j += 1
                if j == len(lines):
                    print("❌ 没有找到__all__的结束符 ']'，请检查 __init__.py 文件结构！")
                    return
                # 在']'前添加
                lines[j] = lines[j].replace(']', "    'SimAM',\n]")
                break

        # 写回文件
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print("✅ SimAM已注册到__init__.py")
    else:
        print("ℹ️  SimAM已经注册")

    # 3. 修改tasks.py添加SimAM到解析器
    print("\n[步骤3] 添加SimAM到模型解析器...")

    tasks_file = ultralytics_path / "nn" / "tasks.py"

    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks_content = f.read()

    if "'SimAM'" not in tasks_content:
        # 找到parse_model函数中的模块字典
        lines = tasks_content.split('\n')

        for i, line in enumerate(lines):
            if "elif m in {" in line and "Conv" in line:
                # 找到模块注册的位置
                j = i
                while j < len(lines) and '}:' not in lines[j]:
                    j += 1
                if j == len(lines):
                    print("❌ 没有找到模块字典的结束符 '}:'，请检查 tasks.py 文件结构！")
                    return
                # 在}前添加
                lines[j - 1] += ", 'SimAM'"
                break

        with open(tasks_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print("✅ SimAM已添加到模型解析器")
    else:
        print("ℹ️  SimAM已在解析器中")

    print("\n" + "=" * 70)
    print("✅ SimAM集成完成！")
    print("=" * 70)
    print("\n现在可以在YAML配置中使用SimAM:")
    print("  - [-1, 1, SimAM, []]")
    print("=" * 70)


if __name__ == "__main__":
    integrate_simam_to_yolov8()