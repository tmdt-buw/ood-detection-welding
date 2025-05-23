from torch import nn
import torch
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LatentEmbeddingCond(nn.Module):

    def __init__(self, input_size: int, d_model: int, cond_size: int) -> None:
        super().__init__()
        self.positional_embedding = PositionalEmbedding(
            d_model=d_model, max_len=input_size)
        self.latent_embedding = nn.Embedding(
            num_embeddings=input_size, embedding_dim=d_model)
        self.cond_embedding = nn.Embedding(
            num_embeddings=cond_size, embedding_dim=d_model)

    def forward(self, x, cond):
        seq_len = x.shape[1]
        x_embed = self.latent_embedding(x) + self.positional_embedding(x)
        cond = self.cond_embedding(cond)
        cond = cond.unsqueeze(1).repeat(1, seq_len, 1)
        return x_embed + cond
    
class LatentEmbedding(nn.Module):

    def __init__(self, input_size: int, d_model: int, seq_len: int = 512) -> None:
        super().__init__()
        self.positional_embedding = PositionalEmbedding(
            d_model=d_model, max_len=seq_len)
        self.latent_embedding = nn.Embedding(
            num_embeddings=input_size, embedding_dim=d_model)
        self.input_size = input_size
        self.d_model = d_model
        self.seq_len = seq_len

    def forward(self, x):
        x_embed = self.latent_embedding(x) + self.positional_embedding(x)
        return x_embed 
