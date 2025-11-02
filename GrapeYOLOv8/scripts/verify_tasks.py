import sys
sys.path.insert(0, 'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8')

# 检查 tasks.py 的位置
import ultralytics.nn.tasks as tasks_module
print(f"tasks.py 文件路径: {tasks_module.__file__}")

# 检查 SimAM 是否正确导入
from ultralytics.nn.modules import SimAM
print(f"SimAM 类: {SimAM}")

# 读取 tasks.py 源代码，检查 SimAM 处理
import inspect
source = inspect.getsource(tasks_module.parse_model)

if "elif m is SimAM:" in source:
    # 提取相关行
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'elif m is SimAM:' in line:
            print(f"\n✅ 找到 SimAM 处理代码（第{i}行）:")
            # 打印前后5行
            for j in range(max(0, i-2), min(len(lines), i+5)):
                print(f"  {lines[j]}")
            break
else:
    print("\n❌ 未找到 SimAM 处理代码！")