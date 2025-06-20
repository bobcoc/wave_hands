import os

# 1. 定义图片和标签目录
png_dir = os.path.join('yolo_dataset', 'png')
txt_dir = os.path.join('yolo_dataset', 'txt')

# 2. 获取所有图片文件名（去除扩展名）
image_names = set()
for fname in os.listdir(png_dir):
    if fname.lower().endswith('.jpg'):
        image_names.add(os.path.splitext(fname)[0])

# 3. 获取所有txt文件名（去除扩展名）
txt_names = set()
for fname in os.listdir(txt_dir):
    if fname.lower().endswith('.txt'):
        txt_names.add(os.path.splitext(fname)[0])

# 4. 找出不匹配的文件
images_without_txt = sorted(image_names - txt_names)
txts_without_image = sorted(txt_names - image_names)

# 5. 输出结果
print('图片无对应txt的文件:')
for name in images_without_txt:
    print(f'  {name}.jpg')
if not images_without_txt:
    print('  无')

print('\ntxt无对应图片的文件:')
for name in txts_without_image:
    print(f'  {name}.txt')
if not txts_without_image:
    print('  无')

if __name__ == "__main__":
    pass  # 逻辑已在脚本顶层实现，无需额外操作 