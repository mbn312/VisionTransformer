from dataclasses import dataclass

@dataclass
class MNISTConfig:
    # Dataset Info
    dataset:str = "mnist"
    n_classes:int = 10
    img_size:tuple[int,int] = (32,32) # (28,28)
    n_channels:int = 1
    # Transformer
    patch_size:tuple[int,int] = (4,4)
    d_model:int = 128
    mlp_hidden:int = 512
    n_heads:int = 8
    n_layers:int = 6
    learned_pe:bool = True
    dropout:float = 0.1
    bias:bool = False
    # Data Augmentation / Normalization
    prob_hflip:float = 0
    crop_padding:int = 0
    train_mean = [0.13066062] if img_size == (28,28) else [0.1309005]
    train_std = [0.30810776] if img_size == (28,28) else [0.2892882]
    # Training
    train_val_split:tuple[int,int] = (50000, 10000)
    get_val_accuracy:bool = True
    batch_size:int = 128
    n_workers:int = 0
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 200
    warmup_epochs:int = 10
    model_location:str = "model.pt"

@dataclass
class FMNISTConfig:
    # Dataset Info
    dataset:str = "fashion_mnist"
    n_classes:int = 10
    img_size:tuple[int,int] = (32,32) # (28,28)
    n_channels:int = 1
    # Transformer
    patch_size:tuple[int,int] = (4,4)
    d_model:int = 128
    mlp_hidden:int = 512
    n_heads:int = 4
    n_layers:int = 6
    learned_pe:bool = True
    dropout:float = 0.1
    bias:bool = False
    # Data Augmentation / Normalization
    prob_hflip:float = 0
    crop_padding:int = 0
    train_mean = [0.2860402] if img_size == (28,28) else [0.2855552]
    train_std = [0.3530239] if img_size == (28,28) else [0.33848408]
    # Training
    train_val_split:tuple[int,int] = (50000, 10000)
    get_val_accuracy:bool = True
    batch_size:int = 128
    n_workers:int = 0
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 200
    warmup_epochs:int = 10
    model_location:str = "model.pt"

@dataclass
class CIFAR10Config:
    # Dataset Info
    dataset:str = "cifar10"
    n_classes:int = 10
    img_size:tuple[int,int] = (32,32)
    n_channels:int = 3
    # Transformer
    patch_size:tuple[int,int] = (4,4)
    d_model:int = 256
    mlp_hidden:int = 1024
    n_heads:int = 8
    n_layers:int = 6
    learned_pe:bool = True
    dropout:float = 0.2
    bias:bool = False
    # Data Augmentation / Normalization
    prob_hflip:float = 0.5
    crop_padding:int = 4
    train_mean = [0.4726623 , 0.47316617, 0.47426093]
    train_std = [0.25135693, 0.2516082 , 0.25173876]
    # Training
    train_val_split:tuple[int,int] = (45000, 5000)
    get_val_accuracy:bool = True
    batch_size:int = 128
    n_workers:int = 0
    lr:float = 5e-4
    lr_min:float = 1e-5
    weight_decay:float = 1e-4
    epochs:int = 200
    warmup_epochs:int = 10
    model_location:str = "model.pt"