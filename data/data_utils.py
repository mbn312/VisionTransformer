import torch
import numpy as np
from data.configs import *
from data.datasets import DatasetSplit
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
CIFAR10.url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

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

def get_mean_std(data, img_channels, denom=1):
    # Get only the images from the dataset
    images = np.array([x[0] for x in data]) / denom

    # Combine pixels of each channel into one dimension
    images = images.reshape(img_channels, -1)

    # Calculate the mean and standard deviation
    mean, std = images.mean(axis=1), images.std(axis=1)

    return mean, std

def get_train_val_split(config):
    if config.dataset == "fashion_mnist":
        dataset = FashionMNIST(root="./../datasets", train=True, download=True, transform=T.Resize(config.img_size))
    elif config.dataset == "cifar10":
        dataset = CIFAR10(root="./../datasets", train=True, download=True, transform=T.Resize(config.img_size))
    else:
        dataset = MNIST(root="./../datasets", train=True, download=True, transform=T.Resize(config.img_size))

    train_set, val_set = torch.utils.data.random_split(dataset, config.train_val_split)
    mean, std = get_mean_std(train_set, config.n_channels, denom=255)

    transform = [
        T.ToTensor(), 
        T.Normalize(mean, std)
    ]

    val_set = DatasetSplit(val_set, classes=dataset.classes if hasattr(dataset, 'classes') else None, transform=T.Compose(transform))

    if config.prob_hflip > 0:
        transform = [T.RandomHorizontalFlip(p=config.prob_hflip)] + transform

    if config.crop_padding != 0:
        transform = [T.RandomCrop(config.img_size[0], padding=config.crop_padding)] + transform

    train_set = DatasetSplit(train_set, classes=dataset.classes if hasattr(dataset, 'classes') else None, transform=T.Compose(transform))

    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size, num_workers=config.n_workers)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=config.batch_size, num_workers=config.n_workers)

    return train_loader, val_loader, mean, std

def get_test_set(config):

    transform = T.Compose([
        T.Resize(config.img_size),
        T.ToTensor(),
        T.Normalize(config.train_mean, config.train_std)
    ])

    if config.dataset == "fashion_mnist":
        test_set = FashionMNIST(train=False, transform=transform)
    elif config.dataset == "cifar10":
        test_set = CIFAR10(train=False, transform=transform)
    else:
        test_set = MNIST(root="./../datasets", train=False, download=True, transform=transform)

    return DataLoader(test_set, shuffle=False, batch_size=config.batch_size, num_workers=config.n_workers)