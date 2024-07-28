import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()

    self.d_model = d_model # Dimensionality of Model
    self.img_size = img_size # Image Size
    self.patch_size = patch_size # Patch Size
    self.n_channels = n_channels # Number of Channels

    self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

  # B: Batch Size
  # C: Image Channels
  # H: Image Height
  # W: Image Width
  # P_col: Patch Column
  # P_row: Patch Row
  def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

    x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

    x = x.transpose(-2, -1) # (B, d_model, P) -> (B, P, d_model)

    return x
  
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length, learned_pe=True, dropout=0.):
    super().__init__()

    if learned_pe:
      positional_encoding = nn.Parameter((d_model ** -0.5) * torch.randn(1, max_seq_length, d_model))
    else:
      positional_encoding = self.create_encoding(max_seq_length, d_model)

    self.register_buffer('positional_encoding', positional_encoding)

    self.dropout = nn.Dropout(dropout)

  def create_encoding(self, max_seq_length, d_model):
    pe = torch.zeros(max_seq_length, d_model)

    for pos in range(max_seq_length):
      for i in range(d_model):
        if i % 2 == 0:
          pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
        else:
          pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

    return pe[None, ...]

  def forward(self, x):
    # Add positional encoding to embeddings
    x = x + self.positional_encoding

    x = self.dropout(x)

    return x 
  
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads, dropout=0.0, bias=False):
    super().__init__()

    self.n_heads = n_heads
    self.head_size = d_model // n_heads
    self.scale = self.head_size ** -0.5
    
    self.query = nn.Linear(d_model, d_model, bias=bias)
    self.key = nn.Linear(d_model, d_model, bias=bias)
    self.value = nn.Linear(d_model, d_model, bias=bias)

    self.output_projection = nn.Linear(d_model, d_model, bias=bias)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, L, d_model = x.shape

    # Obtain query heads
    Q = self.query(x) # (B, L, d_model) -> (B, L, d_model)
    Q = Q.view(B, L, self.n_heads, self.head_size) # (B, L, model_width) -> (B, L, n_heads, head_size)
    Q = Q.transpose(1, 2)  # (B, L, n_heads, head_size) -> (B, n_heads, L, head_size)
    
    # Obtain key heads
    K = self.query(x)
    K = K.view(B, L, self.n_heads, self.head_size)
    K = K.transpose(1, 2)

    # Obtain value heads
    V = self.query(x)
    V = V.view(B, L, self.n_heads, self.head_size)
    V = V.transpose(1, 2) 

    # Get dot product between queries and keys
    attention = torch.matmul(Q, K.transpose(-2, -1))  # (B, n_heads, L, head_size) @ (B, n_heads, head_size, L) -> (B, n_heads, L, L)

    # Scale
    attention = attention * self.scale

    # Apply softmax
    attention = torch.softmax(attention, dim=-1)

    # Get dot product with values
    attention = torch.matmul(attention, V) # (B, n_heads, L, L) @ (B, n_heads, L, head_size) -> (B, n_heads, L, head_size)

    # Combine heads
    attention = attention.transpose(1, 2) # (B, n_heads, L, head_size) -> (B, L, n_heads, head_size)
    attention = attention.contiguous().view(B, L, d_model) # (B, L, n_heads, head_size) -> (B, L, d_model)

    # Output projection
    attention = self.output_projection(attention) # (B, L, d_model) -> (B, L, d_model)

    attention = self.dropout(attention)

    return attention

class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, dropout=0.0, r_mlp=4, bias=False):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads

    # Sub-Layer 1 Normalization
    self.ln1 = nn.LayerNorm(d_model)

    # Multi-Head Attention
    self.mha = MultiHeadAttention(d_model, n_heads, dropout=dropout, bias=bias)

    # Sub-Layer 2 Normalization
    self.ln2 = nn.LayerNorm(d_model)

    # Multilayer Perception
    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model*r_mlp, bias=bias),
        nn.GELU(),
        nn.Linear(d_model*r_mlp, d_model, bias=bias),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    # Residual Connection After Sub-Layer 1
    out = x + self.mha(self.ln1(x))

    # Residual Connection After Sub-Layer 2
    out = out + self.mlp(self.ln2(out))

    return out
  
class VisionTransformer(nn.Module):
  def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, learned_pe=True, dropout=0.0, r_mlp=4, bias=False):
    super().__init__()

    assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    self.d_model = d_model # Dimensionality of model
    self.n_classes = n_classes # Number of classes
    self.img_size = img_size # Image size
    self.patch_size = patch_size # Patch size
    self.n_channels = n_channels # Number of channels
    self.n_heads = n_heads # Number of attention heads

    self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
    self.max_seq_length = self.n_patches + 1
    
    self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)

    self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model)) # Classification Token

    self.positional_encoding = PositionalEncoding( self.d_model, self.max_seq_length, learned_pe=learned_pe, dropout=dropout)

    self.transformer_encoder = nn.Sequential(*[TransformerEncoder( self.d_model, self.n_heads, dropout=dropout, r_mlp=r_mlp, bias=bias) for _ in range(n_layers)])

    # Classification MLP
    self.classifier = nn.Sequential(
        nn.LayerNorm(self.d_model),
        nn.Linear(self.d_model, self.n_classes, bias=bias),
        nn.Softmax(dim=-1)
    )

  def forward(self, images):
    x = self.patch_embedding(images)

    # Expand to have class token for every image in batch
    tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

    # Adding class tokens to the beginning of each embedding
    x = torch.cat((tokens_batch,x), dim=1)

    x = self.positional_encoding(x)

    x = self.transformer_encoder(x)

    x = self.classifier(x[:,0])

    return x