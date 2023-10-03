import torch
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
import os
import numpy as np
from PIL import Image


# 设置文件夹路径和文件数量
folder_path = 'training_set'  # 将 'set' 替换为实际文件夹的路径
num_images = 100

# 创建一个空列表来存储图像数组
big_orig_data_set = []

# 循环读取并处理图像
for i in range(1, num_images + 1):
    # 构建文件路径
    image_path = os.path.join(folder_path, f"{i}.jpg")

    # 打开图像并转换为RGB模式
    img = Image.open(image_path).convert('RGB')

    # 将图像转换为NumPy数组
    img_array = np.array(img)

    # 将图像数组添加到列表中
    big_orig_data_set.append(img_array)

print("图像处理完成，存储在big_orig_data_set中。")


# 设置文件夹路径和文件数量
folder_path = 'small'  # 将 'set' 替换为实际文件夹的路径
num_images = 100

# 创建一个空列表来存储图像数组
small_orig_data_set = []

# 循环读取并处理图像
for i in range(1, num_images + 1):
    # 构建文件路径
    image_path = os.path.join(folder_path, f"{i}.jpg")

    # 打开图像并转换为RGB模式
    img = Image.open(image_path).convert('RGB')

    # 将图像转换为NumPy数组
    img_array = np.array(img)

    # 将图像数组添加到列表中
    small_orig_data_set.append(img_array)

print("图像处理完成，存储在small_orig_data_set中。")


# 创建Generator和Discriminator实例
generator = Generator()
discriminator = Discriminator()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失函数
optimizer_generator = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002)

# 训练循环
num_epochs = 1000  # 训练轮数，根据需要进行调整


for epoch in range(num_epochs):
    # big_orig_data_set中包含真实数据
    for big_orig, small_orig in zip(big_orig_data_set, small_orig_data_set):
        small_orig_tensor = torch.from_numpy(small_orig).float()
        big_orig_tensor = torch.from_numpy(big_orig).float()
        # 训练Discriminator
        optimizer_discriminator.zero_grad()

        # 生成假数据
        big_gen = generator(small_orig_tensor)  # small_orig是Generator的输入

        # 计算真实数据的损失
        real_labels = torch.ones(big_orig_tensor.size(0), 1)  # 真实数据的标签为1
        real_loss = criterion(discriminator(big_orig_tensor), real_labels)

        # 计算假数据的损失
        fake_labels = torch.zeros(big_gen.size(0), 1)  # 假数据的标签为0
        # 注意使用detach()防止梯度传播到Generator
        fake_loss = criterion(discriminator(big_gen.detach()), fake_labels)

        # 总损失为真实损失加上假损失
        total_discriminator_loss = real_loss + fake_loss

        # 反向传播和优化
        total_discriminator_loss.backward()
        optimizer_discriminator.step()

        # 训练Generator
        optimizer_generator.zero_grad()

        # 使用Discriminator来评估假数据
        discriminator_output = discriminator(big_gen)

        # Generator的损失是让Discriminator难以区分生成数据和真实数据
        generator_loss = criterion(discriminator_output, real_labels)

        # 反向传播和优化
        generator_loss.backward()
        optimizer_generator.step()

    # 打印损失等信息来监控训练过程
    print(f'Epoch [{epoch}/{num_epochs}] Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {total_discriminator_loss.item():.4f}')


torch.save(generator, 'generator.pth')
torch.save(discriminator, 'discriminator.pth')
