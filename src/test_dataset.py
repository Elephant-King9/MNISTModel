# 导入数据集
import torchvision

torchvision.datasets.MNIST(root='../datasets', train=False, download=True,
                           transform=torchvision.transforms.ToTensor())
