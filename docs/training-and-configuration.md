# Training and Configuration Reference

This document covers environment setup, the CLI surface in `training.py`, dataset defaults, and practical command examples.

## Environment Setup

Commands below assume you are running from the repository root.

### Python

- The local environment used for this documentation inspection was Python 3.12.4.
- The repository does not ship a `pyproject.toml`, lockfile, or test matrix, so treat the pinned dependencies as the closest thing to an environment contract.

### Install Dependencies

The pinned requirements expect CUDA 12.1 wheels for PyTorch:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

For CPU-only systems:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchvision==0.17.0
pip install datasets==2.17.1 numpy==2.0.0
```

## Data and Artifacts

- Training datasets are downloaded to `../datasets` relative to the current working directory.
- The default checkpoint path is `model.pt`.
- The script expects to be run from a location where those relative paths are acceptable.

## Supported Datasets

Dataset selection is controlled by `--dataset` and maps to presets in [`data/configs.py`](../data/configs.py).

| Dataset | CLI value | Classes | Image size | Channels |
| --- | --- | --- | --- | --- |
| MNIST | `mnist` | 10 | `(28, 28)` | 1 |
| FashionMNIST | `fashion_mnist` | 10 | `(28, 28)` | 1 |
| CIFAR-10 | `cifar10` | 10 | `(32, 32)` | 3 |

## Default Hyperparameters

### MNIST

| Setting | Default |
| --- | --- |
| `patch_size` | `(4, 4)` |
| `d_model` | `128` |
| `mlp_hidden` | `512` |
| `n_heads` | `8` |
| `n_layers` | `6` |
| `dropout` | `0.1` |
| `batch_size` | `128` |
| `lr` | `5e-4` |
| `weight_decay` | `1e-4` |
| `epochs` | `100` |
| `warmup_epochs` | `10` |

### FashionMNIST

| Setting | Default |
| --- | --- |
| `patch_size` | `(4, 4)` |
| `d_model` | `128` |
| `mlp_hidden` | `512` |
| `n_heads` | `4` |
| `n_layers` | `6` |
| `dropout` | `0.1` |
| `batch_size` | `128` |
| `lr` | `5e-4` |
| `weight_decay` | `1e-4` |
| `epochs` | `200` |
| `warmup_epochs` | `10` |

### CIFAR-10

| Setting | Default |
| --- | --- |
| `patch_size` | `(4, 4)` |
| `d_model` | `256` |
| `mlp_hidden` | `1024` |
| `n_heads` | `8` |
| `n_layers` | `6` |
| `dropout` | `0.2` |
| `prob_hflip` | `0.5` |
| `crop_padding` | `4` |
| `batch_size` | `128` |
| `lr` | `5e-4` |
| `weight_decay` | `1e-4` |
| `epochs` | `200` |
| `warmup_epochs` | `10` |

## CLI Reference

`training.py` exposes the following arguments:

| Flag | Meaning |
| --- | --- |
| `-d`, `--dataset` | Dataset preset: `mnist`, `fashion_mnist`, or `cifar10` |
| `-is`, `--img_size` | Image size override as `height width` |
| `-ps`, `--patch_size` | Patch size override as `height width` |
| `-dm`, `--d_model` | Transformer width |
| `-mh`, `--mlp_hidden` | Hidden width of the encoder MLP |
| `-nh`, `--heads` | Number of attention heads |
| `-l`, `--layers` | Number of encoder layers |
| `-lp`, `--learned_pe` | Toggle learned positional encodings |
| `-do`, `--dropout` | Dropout rate |
| `-b`, `--bias` | Toggle bias in linear layers |
| `-ph`, `--prob_hflip` | Horizontal flip probability |
| `-cp`, `--crop_padding` | Random crop padding |
| `-tv`, `--train_val_split` | Train and validation lengths |
| `-va`, `--get_val_accuracy` | Whether to print validation accuracy |
| `-bs`, `--batch_size` | Batch size |
| `-w`, `--workers` | Number of dataloader workers |
| `-lr`, `--lr` | Initial learning rate |
| `-lm`, `--lr_min` | Minimum learning rate for cosine decay |
| `-wd`, `--weight_decay` | Optimizer weight decay |
| `-e`, `--epochs` | Number of training epochs |
| `-we`, `--warmup_epochs` | Number of warmup epochs |
| `-ml`, `--model_location` | Checkpoint output path |

## Example Commands

Train MNIST with defaults:

```bash
python3 training.py --dataset mnist
```

Resize MNIST to 32x32 and use larger 8x8 patches:

```bash
python3 training.py --dataset mnist --img_size 32 32 --patch_size 8 8
```

Train FashionMNIST with a validation split:

```bash
python3 training.py --dataset fashion_mnist --train_val_split 55000 5000 --get_val_accuracy True
```

Train CIFAR-10 with fewer epochs and save to a custom path:

```bash
python3 training.py --dataset cifar10 --epochs 100 --model_location checkpoints/cifar10_vit.pt
```

Train with custom model width and head count:

```bash
python3 training.py --dataset mnist --d_model 256 --heads 8 --layers 8 --mlp_hidden 1024
```

## Validation and Checkpointing

- Validation only runs when the validation split length is greater than zero.
- The best checkpoint is selected by validation loss.
- Without a validation set, the best checkpoint is selected by training loss.
- Parent directories for `--model_location` are created automatically before saving.

## Constraints and Caveats

- `img_size` must be divisible by `patch_size`.
- `img_size` and `patch_size` values must be positive.
- `d_model` must be divisible by `n_heads`.
- `train_val_split` must sum to the dataset length, and the training split must stay positive.
- `warmup_epochs` cannot exceed `epochs`.
- Boolean flags accept common true/false strings such as `True`, `False`, `yes`, `no`, `1`, and `0`.
- The script trains and immediately evaluates; there is no separate `train` versus `eval` subcommand split.
