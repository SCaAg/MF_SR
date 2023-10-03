from PIL import Image
import numpy as np

# 打开图片
image = Image.open('test.jpg')

# 将图片转换为RGB模式（如果它不是RGB模式）
image = image.convert('RGB')

# 将图片转换为NumPy数组
image_array = np.array(image)

# 将RGB888格式转换为RGB444格式
rgb444_image_array = np.round(image_array / 16).clip(0, 15).astype(np.uint8)

# 重新转为rgb888
rgb888_image_array = rgb444_image_array * 16

# 创建一个Pillow图像对象
image = Image.fromarray(rgb888_image_array)

# 保存图像为rst.jpg
image.save('rst.jpg')

print(rgb444_image_array.shape)
