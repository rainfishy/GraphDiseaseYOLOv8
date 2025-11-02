import os

base_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

print("当前数据集规模：")
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(base_dir, 'images', split)
    count = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    print(f"  {split:5s}: {count:4d} 张")

total = sum([len([f for f in os.listdir(os.path.join(base_dir, 'images', s))
                  if f.lower().endswith(('.jpg', '.png'))])
             for s in ['train', 'val', 'test']])
print(f"  总计: {total:4d} 张")