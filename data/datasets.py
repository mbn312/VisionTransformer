import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import Dataset

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
        image = self.dataset[self.split][i]["image"]
        image = self.transform(image)

        label = self.dataset[self.split][i]["label"]

        return image, label
    
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

        self.classes = {
            0: "airplane", 
            1: "automobile", 
            2: "bird", 
            3: "cat", 
            4: "deer", 
            5: "dog", 
            6: "frog", 
            7: "horse", 
            8: "ship", 
            9: "truck"
        }

    def __len__(self):
        return self.dataset.num_rows[self.split]

    def __getitem__(self,i):
        image = self.dataset[self.split][i]["img"]
        image = self.transform(image)

        label = self.dataset[self.split][i]["label"]

        return image, label
    
class DatasetSplit(Dataset):
    def __init__(self, data, classes=None, transform=T.Compose([])):
        self.dataset = data

        self.classes = classes

        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        image = self.transform(self.dataset[i][0])
        label = self.dataset[i][1]

        return image, label