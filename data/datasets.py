import torchvision.transforms as T
from torch.utils.data import Dataset
    
class DatasetSplit(Dataset):
    def __init__(self, data, classes=None, transform=None):
        self.dataset = data

        self.classes = classes

        self.transform = transform or T.Compose([])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        image = self.transform(self.dataset[i][0])
        label = self.dataset[i][1]

        return image, label
