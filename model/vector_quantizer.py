import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import distributed as dist
from vector_quantize_pytorch import ResidualVQ


class ResidualVQLightning(pl.LightningModule):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 0,
        threshold_ema_dead_code: int = 2,
        num_quantizers: int = 1,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.num_quantizers = num_quantizers

        self.vq = ResidualVQ(
            num_quantizers=num_quantizers,
            dim=e_dim,
            codebook_size=n_e,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

        self.save_hyperparameters()

    def forward(self, x):
        """
        Forward pass of the VQ

        Args:
            x: (B, seq_len, embed_dim) input tensor

        Returns:
            z_q: (B, seq_len, embed_dim) quantized output tensor
            loss: (1) scalar tensor
            indices (B, seq_len) indices of z_q
        """
        z_q, indices, commit_loss = self.vq(x)
        # return loss, z_q, perplexity, min_encodings, min_encoding_indices
        return commit_loss, z_q, None, None, indices

    def forward_ood(self, x):
        """
        Forward pass of the VQ with OOD loss

        Args:
            x: (B, seq_len, embed_dim) input tensor

        Returns:
            loss_OOD: (B) OOD loss
            z_q: (B, seq_len, embed_dim) quantized output tensor
            indices (B, seq_len) indices of z_q
            embedding_loss: (1) scalar tensor
        """
        z_q, indices, commit_loss = self.vq(x)
        loss_OOD = torch.mean((z_q.detach() - x) ** 2, dim=[1, 2])
        return loss_OOD, z_q, indices, commit_loss


class VectorQuantizer(pl.LightningModule):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, embed_dim, num_embeddings_per_seq) and flatten
        # print(z.shape)
        z_flattened = z.reshape(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # print("zflattend", z_flattened.shape)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # print("min_encoding_indices", min_encoding_indices.shape)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(
            self.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # print("embedding", self.embedding.weight.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.contiguous()
        return loss, z_q, perplexity, min_encodings, min_encoding_indices

    def get_embedding_from_one_hot(self, min_encoding_indices, target_shape):

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(
            self.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(target_shape)
        z_q = z_q.contiguous()
        return z_q


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def all_reduce(tensor, op=dist.ReduceOp.SUM):

    world_size = get_world_size()

    if world_size == 1:
        return tensor

    dist.all_reduce(tensor, op=op)
