import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from data.datasets import FashionMNIST, CIFAR10
from data.configs import *

def get_config(dataset):
    if dataset == "mnist":
        config = MNISTConfig
    elif dataset == "fashion_mnist":
        config = FMNISTConfig
    elif dataset == "cifar10":
        config = CIFAR10Config
    else:
        raise Exception("Dataset not implemented.")
    
    return config

def get_dataset(config, train=True):

    transform = T.Compose([
        T.Resize(config.img_size),
        T.ToTensor()
    ])

    if config.dataset == "fashion_mnist":
        dataset = FashionMNIST(train=train, transform=transform)
    elif config.dataset == "cifar10":
        dataset = CIFAR10(train=train, transform=transform)
    else:
        dataset = MNIST(root="./../datasets", train=train, download=True, transform=transform)

    return DataLoader(dataset, shuffle=train, batch_size=config.batch_size, num_workers=config.n_workers)