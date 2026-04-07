# Tutorial Drift and Implementation Notes

The README links to two external materials:

- A Google Colab notebook containing the original tutorial code
- A Medium article published on April 4, 2024

Those resources are useful for understanding the original learning-oriented version of the project, but they do not fully describe the current repository state.

## What the Tutorial Materials Cover

The notebook and article focus on:

- Building a Vision Transformer from scratch in PyTorch
- Explaining patch embeddings, positional encoding, attention, encoder blocks, and the classifier step by step
- Training an MNIST-only model with hardcoded hyperparameters inside the notebook
- Demonstrating an MNIST result of about 92% accuracy after 5 epochs

The notebook currently contains 20 cells and mirrors the article's tutorial flow closely.

## What the Current Repo Adds

Compared with the tutorial materials, the current codebase adds:

- A modular file layout instead of a single notebook-centric implementation
- Config dataclasses for MNIST, FashionMNIST, and CIFAR-10
- CLI overrides through `argparse`
- Optional train/validation splits
- Validation loss reporting and optional validation accuracy reporting
- Data augmentation and normalization helpers
- Warmup plus cosine learning-rate scheduling
- Conditional use of `Adam` versus `AdamW`
- Checkpoint saving and reload-based test evaluation
- Optional dropout, bias toggles, and a learned-versus-sinusoidal positional encoding switch

## How to Read the Project Today

Use the resources in this order:

1. Read the tutorial article or notebook if you want the conceptual story.
2. Read [`training.py`](../training.py) if you want the actual runtime entrypoint.
3. Read [`models/model.py`](../models/model.py) and [`data/data_utils.py`](../data/data_utils.py) for implementation truth.

In short:

- The tutorial explains the intended ideas.
- The repository code defines the behavior you will actually run.

## Implementation Notes That Matter

These are the most important code-level observations for anyone using or extending the repo.

### The Current Code Is More Production-Oriented Than the Tutorial

The repository code now includes several implementation details that go beyond the original notebook:

- Learned positional encodings are trainable parameters when `learned_pe=True`.
- Attention uses distinct query, key, and value projections.
- The classifier returns raw logits, which matches `nn.CrossEntropyLoss()`.
- Checkpoint parent directories are created automatically before saving.
- CLI boolean flags accept common true/false strings instead of relying on Python's `bool(...)` coercion.

### Validation-Free Runs Still Save Checkpoints

When there is no validation split, the training loop falls back to training loss for checkpoint selection.

This keeps validation-free runs from silently defaulting to "last epoch wins."

## Documentation Stance

This repository's documentation uses the following rule:

- External tutorial materials are referenced as educational background.
- The current Python source code is treated as the source of truth for setup and behavior.
- Where the code and tutorial disagree, the docs call that out instead of smoothing it over.
