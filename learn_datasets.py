import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from torch import nn

trans = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
train_set = torchvision.datasets.CIFAR10("./datasets", True, transform=trans, download=True)
test_set = torchvision.datasets.CIFAR10("./datasets", False, transform=trans, download=True)
dataloader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=2, drop_last=True)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = Model()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    img, target = data
    output = model(img)
    writer.add_images("input", img, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1
    print(output.shape)
writer.close()
