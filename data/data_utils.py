import torch
import numpy as np
from data.configs import *
from data.datasets import DatasetSplit
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

def get_config(args):
    if args.dataset == "mnist":
        config = MNISTConfig
    elif args.dataset == "fashion_mnist":
        config = FMNISTConfig
    elif args.dataset == "cifar10":
        CIFAR10.url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        config = CIFAR10Config
    else:
        raise Exception("Dataset not implemented.")
    
    if args.img_size is not None:
        config.img_size = (args.img_size[0], args.img_size[0]) if len(args.img_size) == 1 else (args.img_size[0], args.img_size[1])

    if args.patch_size is not None:
        config.patch_size = (args.patch_size[0], args.patch_size[0]) if len(args.patch_size) == 1 else (args.patch_size[0], args.patch_size[1])

    assert config.img_size[0] % config.patch_size[0] == 0 and config.img_size[1] % config.patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"

    if args.d_model is not None:
        config.d_model = args.d_model

    if args.mlp_hidden is not None:
        config.mlp_hidden = args.mlp_hidden

    if args.heads is not None:
        config.n_heads = args.heads

    assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

    if args.layers is not None:
        config.n_layers = args.layers

    if args.learned_pe is not None:
        config.learned_pe = args.learned_pe

    if args.dropout is not None:
        config.dropout = args.dropout

    if args.bias is not None:
        config.bias = args.bias

    if args.prob_hflip is not None:
        config.prob_hflip = args.prob_hflip

    if args.crop_padding is not None:
        config.crop_padding = args.crop_padding

    if args.get_val_accuracy is not None:
        config.get_val_accuracy = args.get_val_accuracy

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.workers is not None:
        config.n_workers = args.workers

    if args.lr is not None:
        config.lr = args.lr

    if args.lr_min is not None:
        config.lr_min = args.lr_min

    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay

    if args.epochs is not None:
        config.epochs = args.epochs

    if args.warmup_epochs is not None:
        config.warmup_epochs = args.warmup_epochs

    assert config.epochs >= config.warmup_epochs, "Warmup epochs cannot be higher than number than epochs"

    if args.model_location is not None:
        config.model_location = args.model_location

    if config.dataset == "fashion_mnist":
        train_set = FashionMNIST(root="./../datasets", train=True, download=True, transform=T.Resize(config.img_size))
    elif config.dataset == "cifar10":
        train_set = CIFAR10(root="./../datasets", train=True, download=True, transform=T.Resize(config.img_size))
    else:
        train_set = MNIST(root="./../datasets", train=True, download=True, transform=T.Resize(config.img_size))

    if args.train_val_split is not None:
        config.train_val_split = (args.train_val_split[0], len(train_set) - args.train_val_split[0]) if len(args.train_val_split) == 1 else (args.train_val_split[0], args.train_val_split[1]) 

    assert ((config.train_val_split[0] + config.train_val_split[1]) == len(train_set)) and (config.train_val_split[0] >= 0) and (config.train_val_split[1] >= 0), "Sum of splits must be equal to length of training data"

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
        dataset = FashionMNIST(root="./../datasets", train=True, download=False, transform=T.Resize(config.img_size))
    elif config.dataset == "cifar10":
        dataset = CIFAR10(root="./../datasets", train=True, download=False, transform=T.Resize(config.img_size))
    else:
        dataset = MNIST(root="./../datasets", train=True, download=False, transform=T.Resize(config.img_size))

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
        test_set = FashionMNIST(root="./../datasets", train=False, download=False, transform=transform)
    elif config.dataset == "cifar10":
        test_set = CIFAR10(root="./../datasets", train=False, download=False, transform=transform)
    else:
        test_set = MNIST(root="./../datasets", train=False, download=False, transform=transform)

    return DataLoader(test_set, shuffle=False, batch_size=config.batch_size, num_workers=config.n_workers)