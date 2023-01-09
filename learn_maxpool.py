import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d

# example of numbers tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input_ten = torch.tensor([[1, 2, 0, 3, 1],
                          [0, 1, 2, 3, 1],
                          [1, 2, 1, 0, 0],
                          [5, 2, 3, 1, 1],
                          [2, 1, 0, 1, 1]], dtype=torch.float64)

input_ten = torch.reshape(input_ten, (-1, 1, 5, 5))

# example of images tensor

datasets = torchvision.datasets.CIFAR10("datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
dataloader = DataLoader(datasets, batch_size=64, shuffle=True, num_workers=2, drop_last=True)


class ModelMaxPool(nn.Module):
    def __init__(self):
        super(ModelMaxPool, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x1 = self.maxpool1(x)
        return x1


model = ModelMaxPool()
step = 0
writer = SummaryWriter("logs")
for data in dataloader:
    img, target = data
    print(img.shape)
    writer.add_images("input", img, step)
    writer.add_images("output", model(img), step)
    step = step + 1
writer.close()
