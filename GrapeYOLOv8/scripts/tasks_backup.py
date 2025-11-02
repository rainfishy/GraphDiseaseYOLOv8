# 创建备份脚本 backup_tasks.py
import shutil
import os

source = 'C:/Users/zjl04/.conda/envs/GrapeDiseaseYOLOv8/lib/site-packages/ultralytics/nn/tasks.py'
backup = 'C:/Users/zjl04/.conda/envs/GrapeDiseaseYOLOv8/lib/site-packages/ultralytics/nn/tasks.py.backup'

if os.path.exists(source):
    shutil.copy2(source, backup)
    print(f"✅ 已备份: {backup}")
else:
    print("❌ 源文件不存在")