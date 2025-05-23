# https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here.
    """

    def __init__(self, d_model, seq_len, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert d_model % n_head == 0
        
        # Initialize with larger values for better gradient flow
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)
        
        # Rest of the initialization remains same
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len),
        )
        self.n_head = n_head
        self.n_embd = d_model
        self.last_attn_weights = None

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Store both raw and normalized attention
        self.last_attn_weights = {
            'raw': att.detach(),
            'normalized': F.softmax(att, dim=-1).detach()
        }
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, d_model, seq_len, n_head, res_dropout, att_dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, seq_len, n_head, att_dropout, res_dropout
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(d_model, 4 * d_model),
                c_proj=nn.Linear(4 * d_model, d_model),
                act=NewGELUActivation(),
                dropout=nn.Dropout(res_dropout),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
