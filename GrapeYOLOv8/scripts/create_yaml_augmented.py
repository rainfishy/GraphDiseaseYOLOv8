"""
创建增强数据集的YAML配置文件
"""

import os
import yaml


def create_yaml_config():
    """创建YAML配置文件"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'data_augmented')

    # 数据集配置
    config = {
        'path': '../data_augmented',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 4,
        'names': {
            0: 'black_rot',
            1: 'blight',
            2: 'black_measles',
            3: 'Healthy'
        }
    }

    # 保存路径
    yaml_path = os.path.join(output_dir, 'grape_augmented.yaml')

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 写入YAML
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print("=" * 70)
    print("✅ YAML配置文件创建成功！")
    print("=" * 70)
    print(f"保存路径: {yaml_path}")
    print("\n配置内容:")
    print("-" * 70)

    with open(yaml_path, 'r', encoding='utf-8') as f:
        print(f.read())

    print("=" * 70)


if __name__ == "__main__":
    create_yaml_config()