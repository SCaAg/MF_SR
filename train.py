import torch
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator

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
    for real_data in dataloader:  # dataloader中包含真实数据
        # 训练Discriminator
        optimizer_discriminator.zero_grad()

        # 生成假数据
        fake_data = generator(noise)  # 假设noise是Generator的输入

        # 计算真实数据的损失
        real_labels = torch.ones(real_data.size(0), 1)  # 真实数据的标签为1
        real_loss = criterion(discriminator(real_data), real_labels)

        # 计算假数据的损失
        fake_labels = torch.zeros(fake_data.size(0), 1)  # 假数据的标签为0
        # 注意使用detach()防止梯度传播到Generator
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)

        # 总损失为真实损失加上假损失
        total_discriminator_loss = real_loss + fake_loss

        # 反向传播和优化
        total_discriminator_loss.backward()
        optimizer_discriminator.step()

        # 训练Generator
        optimizer_generator.zero_grad()

        # 生成假数据
        fake_data = generator(noise)

        # 使用Discriminator来评估假数据
        discriminator_output = discriminator(fake_data)

        # Generator的损失是让Discriminator难以区分生成数据和真实数据
        generator_loss = criterion(discriminator_output, real_labels)

        # 反向传播和优化
        generator_loss.backward()
        optimizer_generator.step()

    # 打印损失等信息来监控训练过程
    print(f'Epoch [{epoch}/{num_epochs}] Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {total_discriminator_loss.item():.4f}')
