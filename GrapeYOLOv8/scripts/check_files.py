import os

print("当前工作目录:", os.getcwd())
print("\n检查文件是否存在:")

# 检查可能的路径
paths = [
    'GrapeYOLOv8/models/yolov8n_simam.yaml',
    'models/yolov8n_simam.yaml',
    'E:/YOLOGrape/Grape_Disease_Experiment/GrapeDiseaseYOLOv8/GrapeYOLOv8/models/yolov8n_simam.yaml',
]

for path in paths:
    exists = os.path.exists(path)
    print(f"  {'✅' if exists else '❌'} {path}")

# 列出 models 目录下的所有文件
models_dir = 'GrapeYOLOv8/models'
if os.path.exists(models_dir):
    print(f"\n📁 {models_dir} 目录下的文件:")
    for file in os.listdir(models_dir):
        print(f"  - {file}")