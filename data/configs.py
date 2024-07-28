from dataclasses import dataclass

@dataclass
class MNISTConfig:
    dataset = "mnist"
    n_classes = 10
    img_size = (32,32)
    n_channels = 1
    n_classes = 10
    d_model = 9
    patch_size = (16,16)
    n_heads = 3
    n_layers = 3
    learned_pe = True
    dropout = 0.0
    r_mlp = 4
    bias = False
    batch_size = 128
    n_workers = 0
    epochs = 5
    lr = 0.005