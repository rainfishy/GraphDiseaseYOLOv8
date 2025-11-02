import shutil
import os


def clean_augmented_data():
    """清理之前生成的增强数据"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    augmented_dir = os.path.join(current_dir, '..', 'data_augmented')

    print("=" * 70)
    print("🧹 清理之前的增强数据")
    print("=" * 70)

    if os.path.exists(augmented_dir):
        print(f"\n📂 找到目录: {augmented_dir}")

        # 统计文件数
        total_files = sum([len(files) for _, _, files in os.walk(augmented_dir)])
        print(f"📊 包含文件数: {total_files}")

        # 确认删除
        print("\n⚠️  即将删除该目录及所有内容")
        confirm = input("确认删除? (输入 'yes' 继续): ")

        if confirm.lower() == 'yes':
            shutil.rmtree(augmented_dir)
            print("✅ 已删除")
        else:
            print("❌ 取消删除")
    else:
        print("✅ 目录不存在，无需清理")

    print("=" * 70)


if __name__ == "__main__":
    clean_augmented_data()