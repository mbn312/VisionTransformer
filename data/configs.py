from dataclasses import dataclass

@dataclass
class MNISTConfig:
    # Dataset Info
    dataset = "mnist"
    n_classes = 10
    img_size = (28,28)
    n_channels = 1
    n_classes = 10
    # Transformer
    d_model = 256
    patch_size = (4,4)
    n_heads = 8
    n_layers = 6
    learned_pe = True
    dropout = 0.2
    r_mlp = 4
    bias = False
    # Data Augmentation / Normalization
    prob_hflip = 0.5
    crop_padding = 4
    train_mean = [0.13066062]
    train_std = [0.30810776]
    # Training
    train_val_split = (50000, 10000)
    get_val_accuracy = True
    batch_size = 128
    n_workers = 0
    lr = 5e-4
    lr_min = 1e-5
    weight_decay = 1e-4
    epochs = 200
    warmup_epochs = 10
    model_location = "model.pt"

@dataclass
class FMNISTConfig:
    # Dataset Info
    dataset = "fashion_mnist"
    n_classes = 10
    img_size = (28,28)
    n_channels = 1
    n_classes = 10
    # Transformer
    d_model = 256
    patch_size = (4,4)
    n_heads = 8
    n_layers = 6
    learned_pe = True
    dropout = 0.2
    r_mlp = 4
    bias = False
    # Data Augmentation / Normalization
    prob_hflip = 0.5
    crop_padding = 4
    train_mean = [0.2860402]
    train_std = [0.3530239]
    # Training
    train_val_split = (50000, 10000)
    get_val_accuracy = True
    batch_size = 128
    n_workers = 0
    lr = 5e-4
    lr_min = 1e-5
    weight_decay = 1e-4
    epochs = 200
    warmup_epochs = 10
    model_location = "model.pt"

@dataclass
class CIFAR10Config:
    # Dataset Info
    dataset = "cifar10"
    n_classes = 10
    img_size = (32,32)
    n_channels = 3
    n_classes = 10
    # Transformer
    d_model = 256
    patch_size = (4,4)
    n_heads = 8
    n_layers = 6
    learned_pe = True
    dropout = 0.2
    r_mlp = 4
    bias = False
    # Data Augmentation / Normalization
    prob_hflip = 0.5
    crop_padding = 4
    train_mean = [0.4726623 , 0.47316617, 0.47426093]
    train_std = [0.25135693, 0.2516082 , 0.25173876]
    # Training
    train_val_split = (45000, 5000)
    get_val_accuracy = True
    batch_size = 128
    n_workers = 0
    lr = 5e-4
    lr_min = 1e-5
    weight_decay = 1e-4
    epochs = 200
    warmup_epochs = 10
    model_location = "model.pt"