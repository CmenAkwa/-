from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "dataset/train/bees/16838648_415acd9e3f.jpg"
img = Image.open(img_path)

print(type(img))

writer = SummaryWriter('transforms_logs')

# transforms使用
trans_tensor = transforms.ToTensor()  # 创建对象创建对象，一个转化的工具
tensor_img = trans_tensor(img)  # 把PIL类型变成张量img
# print(tensor_img)  # 转为张量tensor数组，tensor内存储了一些神经网络的参数
print(tensor_img.shape)  # WHC,宽高通

writer.add_image('tensor1', tensor_img, 0)
# 归一化
trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(img_norm.shape)
writer.add_image('norm', img_norm, 1)

# resise 两个参数任意缩放，一个参数等比缩放
print(img.size)
trans_resize = transforms.Resize((512, 512))  # 定义工具
# PIL->resize PIL
img_resize = trans_resize(img)  # 使用
print(img_resize)
# resize PIL->tensor,覆盖原来的版本，改了格式
img_resize = trans_tensor(img_resize)
writer.add_image('resizeasd', img_resize, 5)

# Compose 组合操作，创建流水线
trans_resize_2 = transforms.Resize(512)
# PIL->PIL>tensor 格式变换流程
trans_compose = transforms.Compose([trans_resize_2, trans_tensor])
img_resize_2 = trans_compose(img)
writer.add_image("compose", img_resize_2, 3)
writer.close()
