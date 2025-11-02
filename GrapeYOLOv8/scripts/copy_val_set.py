"""
复制验证集到增强数据目录
"""

import os
import shutil
from tqdm import tqdm


def copy_val_set():
    """复制验证集"""

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 源目录和目标目录
    src_base = os.path.join(current_dir, '..', 'data')
    dst_base = os.path.join(current_dir, '..', 'data_augmented')

    print("=" * 70)
    print("📋 复制验证集到增强数据目录")
    print("=" * 70)

    # 复制images和labels
    for subdir in ['images', 'labels']:
        src_dir = os.path.join(src_base, subdir, 'val')
        dst_dir = os.path.join(dst_base, subdir, 'val')

        if not os.path.exists(src_dir):
            print(f"❌ 源目录不存在: {src_dir}")
            continue

        # 创建目标目录
        os.makedirs(dst_dir, exist_ok=True)

        # 复制文件
        files = os.listdir(src_dir)
        print(f"\n复制 {subdir}/val...")

        for file in tqdm(files, desc=f"  进度", ncols=70):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)

        print(f"✅ 完成: {len(files)} 个文件")

    print("\n" + "=" * 70)
    print("✅ 验证集复制完成！")
    print("=" * 70)


if __name__ == "__main__":
    copy_val_set()