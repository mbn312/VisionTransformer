import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset


class FashionMNIST(Dataset):
    def __init__(self, train=True, img_size=(28,28), transform=None):
        self.dataset = load_dataset("fashion_mnist", trust_remote_code=True)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(img_size),
                T.ToTensor()
            ])

        if train:
            self.split = "train"
        else:
            self.split = "test"

        self.classes = {
            0: "t-shirt/top",
            1: "trousers",
            2: "pullover",
            3: "dress",
            4: "coat",
            5: "sandal",
            6: "shirt",
            7: "sneaker",
            8: "bag",
            9: "ankle boot"
        }

    def __len__(self):
        return self.dataset.num_rows[self.split]

    def __getitem__(self, i):
        img = self.dataset[self.split][i]["image"]
        img = self.transform(img)

        label = self.dataset[self.split][i]["label"]

        return img, label
    
class CIFAR10(Dataset):
    def __init__(self, train=True, img_size=(32,32), transform=None):
        self.dataset = load_dataset("cifar10", trust_remote_code=True)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize(img_size),
                T.ToTensor()
            ])

        if train:
            self.split = "train"
        else:
            self.split = "test"

    def __len__(self):
        return self.dataset.num_rows[self.split]

    def __getitem__(self,i):
        img = self.dataset[self.split][i]["img"]
        img = self.transform(img)

        label = self.dataset[self.split][i]["label"]

        return img, label