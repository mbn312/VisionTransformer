import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from data.configs import MNISTConfig

def get_config(dataset):
    if dataset == "mnist":
        config = MNISTConfig
    else:
        raise Exception("Config for this dataset has not been implemented.")
    
    return config

def get_dataset(config, train=True):
    transform = T.Compose([
        T.Resize(config.img_size),
        T.ToTensor()
    ])

    data = MNIST(root="./../datasets", train=train, download=True, transform=transform)

    return DataLoader(data, shuffle=train, batch_size=config.batch_size, num_workers=config.n_workers)