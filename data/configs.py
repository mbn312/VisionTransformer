from dataclasses import dataclass

@dataclass
class MNISTConfig:
    dataset = "mnist"
    n_classes = 10
    img_size = (28,28)
    n_channels = 1
    n_classes = 10
    d_model = 9
    patch_size = (14,14)
    n_heads = 3
    n_layers = 3
    learned_pe = True
    dropout = 0.0
    r_mlp = 4
    bias = False
    train_val_split = (50000, 10000)
    get_val_accuracy = True
    batch_size = 128
    n_workers = 0
    lr = 5e-4
    lr_min = 1e-5
    weight_decay = 1e-4
    epochs = 5
    warmup_epochs = 0
    model_location = "model.pt"

@dataclass
class FMNISTConfig:
    dataset = "fashion_mnist"
    n_classes = 10
    img_size = (28,28)
    n_channels = 1
    n_classes = 10
    d_model = 9
    patch_size = (14,14)
    n_heads = 3
    n_layers = 3
    learned_pe = True
    dropout = 0.0
    r_mlp = 4
    bias = False
    train_val_split = (50000, 10000)
    get_val_accuracy = True
    batch_size = 128
    n_workers = 0
    lr = 5e-4
    lr_min = 1e-5
    weight_decay = 1e-4
    epochs = 5
    warmup_epochs = 0
    model_location = "model.pt"

@dataclass
class CIFAR10Config:
    dataset = "cifar10"
    n_classes = 10
    img_size = (32,32)
    n_channels = 3
    n_classes = 10
    d_model = 9
    patch_size = (16,16)
    n_heads = 3
    n_layers = 3
    learned_pe = True
    dropout = 0.0
    r_mlp = 4
    bias = False
    train_val_split = (45000, 5000)
    get_val_accuracy = True
    batch_size = 128
    n_workers = 0
    lr = 5e-4
    lr_min = 1e-5
    weight_decay = 1e-4
    epochs = 5
    warmup_epochs = 0
    model_location = "model.pt"