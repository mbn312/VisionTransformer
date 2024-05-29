import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

def get_dataset(img_size, batch_size, train=True):
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor()
    ])

    data = MNIST(root="./../datasets", train=train, download=True, transform=transform)

    return DataLoader(data, shuffle=train, batch_size=batch_size)