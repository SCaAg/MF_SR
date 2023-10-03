import os
import shutil

# 设置文件夹路径
folder_path = 'training_set'  # 将 'set' 替换为实际文件夹的路径

# 获取文件夹中的所有 .jpg 文件
jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 重命名文件
start_index = 1

for old_name in jpg_files:
    # 构建新的文件名，例如 "1.jpg", "2.jpg"，以此类推
    new_name = f"{start_index}.jpg"

    # 构建完整的文件路径
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)

    # 重命名文件
    shutil.move(old_path, new_path)

    # 增加计数器
    start_index += 1

print("重命名完成。")
