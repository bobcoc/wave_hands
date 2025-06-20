import os
from pathlib import Path
from tqdm import tqdm

def change_label_class(labels_dir, old_class=0, new_class=1):
    """
    修改标签文件中的类别ID
    
    Args:
        labels_dir: 标签文件目录
        old_class: 原类别ID
        new_class: 新类别ID
    """
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        raise ValueError(f"目录不存在: {labels_dir}")
    
    # 获取所有txt文件
    label_files = list(labels_dir.glob("*.txt"))
    print(f"找到 {len(label_files)} 个标签文件")
    
    # 遍历处理每个文件
    for label_file in tqdm(label_files, desc="处理标签文件"):
        # 读取文件内容
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 修改类别ID
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1 and parts[0] == str(old_class):
                parts[0] = str(new_class)
            new_lines.append(' '.join(parts))
        
        # 写回文件
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

def main():
    labels_dir = "yolo_dataset1/labels"
    try:
        change_label_class(labels_dir)
        print("类别修改完成！")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main() 