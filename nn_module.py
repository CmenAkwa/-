import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class asd(nn.Module):
    def __init__(self):  # 创建结构，构造网络
        super(asd, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)  # 卷积层，彩色图像通道数3

    def forward(self, x):  # 网络运行逻辑
        x = self.conv1(x)  # 进入第一层
        return x


# input_array = torch.tensor([[1, 2, 0, 3, 1],
#                             [0, 1, 2, 3, 1, ],
#                             [1, 2, 1, 0, 0, ],
#                             [5, 2, 3, 1, 1, ],
#                             [2, 1, 0, 1, 1, ]])
#
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
#
# input_array = torch.reshape(input_array, (1, 1, 5, 5))
# kernel = torch.reshape(kernel, (1, 1, 3, 3))
#
# print(input_array.shape)
# print(kernel.shape)
# output_array = F.conv2d(input_array, kernel, stride=1, padding=1)  # padding=n周围添加n圈0，stride=1，卷积核移动的步数为1
# print(output_array)

dataset = torchvision.datasets.CIFAR10(root='./torch_dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())  # 测试集
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)
test = asd()
print(test)

writer = SummaryWriter('test')

step = 0
for data in dataloader:
    imgs, targets = data
    output = test(imgs)
    writer.add_images('imput', imgs, global_step=step)
    output = torch.reshape(output, (-1,3,30,30))
    writer.add_images('output', output, global_step=step)
    step = step + 1


writer.close()