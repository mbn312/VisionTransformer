# Architecture Walkthrough

This document describes how the current repository behaves, starting from `python3 training.py` and ending with a saved checkpoint and test-set accuracy report.

## End-to-End Flow

The main runtime path is:

1. Parse CLI arguments in [`training.py`](../training.py).
2. Build a dataset-specific config in [`data/data_utils.py`](../data/data_utils.py).
3. Create train and validation dataloaders, plus dataset normalization statistics.
4. Instantiate `VisionTransformer` from [`models/model.py`](../models/model.py).
5. Train for `config.epochs`, optionally using warmup and cosine decay.
6. Save the best checkpoint observed during training.
7. Reload the checkpoint and evaluate it on the test set.

## Configuration Selection

`get_config(args)` chooses one of three dataclass presets from [`data/configs.py`](../data/configs.py):

- `MNISTConfig`
- `FMNISTConfig`
- `CIFAR10Config`

Each preset defines:

- Dataset metadata such as image size, number of channels, and number of classes.
- Model hyperparameters such as patch size, `d_model`, MLP width, number of heads, and encoder depth.
- Training defaults such as batch size, learning rate, weight decay, epochs, and checkpoint path.
- Data augmentation defaults for horizontal flips and random crops.

CLI arguments selectively overwrite those defaults before training starts.

## Data Pipeline

The data helpers live in [`data/data_utils.py`](../data/data_utils.py).

### Dataset Loading

- `get_config(args)` downloads the training split if it is missing.
- `get_train_val_split(config)` reloads the training split without downloading, then applies `torch.utils.data.random_split`.
- `get_test_set(config)` loads the test split using the training mean and standard deviation computed from the chosen training subset.

By default, datasets are stored in `../datasets` relative to the current working directory.

### Normalization and Augmentation

The training pipeline computes `mean` and `std` from the sampled training subset via `get_mean_std()` and then applies:

1. `ToTensor()`
2. `Normalize(mean, std)`

Optional augmentations are prepended for the training split only:

- `RandomHorizontalFlip`
- `RandomCrop`

Validation and test data are not augmented.

`get_mean_std()` converts the sampled images to NumPy arrays, normalizes by `denom`, moves the channel axis to the front, flattens each channel, and computes per-channel mean and standard deviation. This now behaves correctly for both grayscale datasets and CIFAR-10.

### Dataset Wrapper

[`data/datasets.py`](../data/datasets.py) defines `DatasetSplit`, a minimal wrapper around the `Subset` objects returned by `random_split`. It applies the final transform pipeline inside `__getitem__`.

## Model Stack

The core model lives in [`models/model.py`](../models/model.py).

### Patch Embedding

`PatchEmbedding` uses a `Conv2d` layer with:

- `kernel_size = patch_size`
- `stride = patch_size`

That converts an image tensor from `(B, C, H, W)` into non-overlapping patch embeddings. The output is flattened and transposed to `(B, P, d_model)`, where `P` is the number of patches.

### Positional Encoding and Class Token

`VisionTransformer` creates:

- `self.cls_token`, a learned classification token
- `self.positional_encoding`, a `PositionalEncoding` module sized to `n_patches + 1`

`PositionalEncoding` supports two modes:

- Learned positional encodings from random initialization
- Sinusoidal positional encodings built with NumPy

At runtime, the class token is concatenated to the front of the patch sequence and positional encodings are added to the full sequence.

### Multi-Head Attention

`MultiHeadAttention` projects tokens into distinct query, key, and value spaces, reshapes them into heads, computes scaled dot-product attention, merges heads back together, and applies an output projection.

The tensor flow is:

1. `(B, L, d_model)` to head-shaped Q/K/V tensors
2. attention scores from `Q @ K^T`
3. scaling by `head_size ** -0.5`
4. `softmax`
5. weighted sum with `V`
6. output projection back to `(B, L, d_model)`

### Transformer Encoder

Each `TransformerEncoder` block contains:

- Pre-norm layer normalization
- Multi-head self-attention
- Residual connection
- Pre-norm layer normalization
- Two-layer MLP with GELU and dropout
- Residual connection

`VisionTransformer` stacks `n_layers` of these blocks inside an `nn.Sequential`.

### Classification Head

After the encoder stack, the model selects the class token at index `0` and applies:

1. `LayerNorm(d_model)`
2. `Linear(d_model, n_classes)`

The output logits are returned directly to the training loop.

## Training Loop

`train_model(config)` in [`training.py`](../training.py) handles optimization.

### Optimizer and Schedulers

- `Adam` is used when `weight_decay == 0`.
- `AdamW` is used otherwise.
- `CosineAnnealingLR` runs after the warmup phase.
- `LinearLR` is created when `warmup_epochs > 0`.

### Epoch Behavior

Each epoch:

1. Trains over the training dataloader.
2. Steps either the warmup or cosine scheduler.
3. Optionally evaluates on the validation set.
4. Saves `model.state_dict()` when validation loss improves.
5. Prints training loss, and optionally validation loss and validation accuracy.

If there is no validation split, the code falls back to training loss for checkpoint selection.

## Evaluation Flow

After training, `get_model_accuracy(config)`:

1. Rebuilds the model from the same config.
2. Loads the saved checkpoint from `config.model_location`.
3. Builds the test dataloader.
4. Reports classification accuracy over the full test set.

The script does not emit confusion matrices, per-class metrics, or experiment metadata.

## Practical Reading Order

If you are new to the repo, read files in this order:

1. [`training.py`](../training.py)
2. [`data/configs.py`](../data/configs.py)
3. [`data/data_utils.py`](../data/data_utils.py)
4. [`models/model.py`](../models/model.py)
5. [`docs/tutorial-drift.md`](tutorial-drift.md)
