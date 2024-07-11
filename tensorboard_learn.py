from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# for i in range(100):
#     writer.add_scalar("y=x", 2*i, i)  # 纵坐标value，横坐标step,画图用的
writer = SummaryWriter("logs")
image_path = "dataset/train/ants/5650366_e22b7e1065.jpg"  # 设置图片路径
img_PIL = Image.open(image_path)  #
img_array = np.array(img_PIL)  # 转为RGB存储每一个图像像素点成为numpy数组
print(img_array.shape)  # 发现是HWC，高宽通

writer.add_image("text", img_array, 1, dataformats='HWC')  # 转为HWC
writer.close()