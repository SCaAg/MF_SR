from PIL import Image
import numpy as np
import os

# 设置文件夹路径和文件数量
folder_path = 'training_set'  # 将 'set' 替换为实际文件夹的路径
num_images = 100

# 创建一个空列表来存储图像数组
image_list = []

# 循环读取并处理图像
for i in range(1, num_images + 1):
    # 构建文件路径
    image_path = os.path.join(folder_path, f"{i}.jpg")

    # 打开图像并转换为RGB模式
    img = Image.open(image_path).convert('RGB')

    # 将图像转换为NumPy数组
    img_array = np.array(img)

    # 将图像数组添加到列表中
    image_list.append(img_array)

print("图像处理完成，存储在image_list中。")
