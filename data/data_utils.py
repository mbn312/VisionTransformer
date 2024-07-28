import torch
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

def get_train_val_split(config):
    transform = T.Compose([
        T.Resize(config.img_size),
        T.ToTensor()
    ])

    if config.dataset == "fashion_mnist":
        dataset = FashionMNIST(train=True, transform=transform)
    elif config.dataset == "cifar10":
        dataset = CIFAR10(train=True, transform=transform)
    else:
        dataset = MNIST(root="./../datasets", train=True, download=True, transform=transform)

    train_set, val_set = torch.utils.data.random_split(dataset, config.train_val_split)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size, num_workers=config.n_workers)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=config.batch_size, num_workers=config.n_workers)

    return train_loader, val_loader

def get_test_set(config):

    transform = T.Compose([
        T.Resize(config.img_size),
        T.ToTensor()
    ])

    if config.dataset == "fashion_mnist":
        test_set = FashionMNIST(train=False, transform=transform)
    elif config.dataset == "cifar10":
        test_set = CIFAR10(train=False, transform=transform)
    else:
        test_set = MNIST(root="./../datasets", train=False, download=True, transform=transform)

    return DataLoader(test_set, shuffle=False, batch_size=config.batch_size, num_workers=config.n_workers)