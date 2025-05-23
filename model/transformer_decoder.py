import torch
import torch.nn as nn
import math
import logging
from torch.nn import functional as F
from model.embedding import LatentEmbedding
from model.transformer_block import Block
from model.transformer_base import TransformerDecoderBase


class MyTransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        dataset_name: str,
        d_model: int = 64,
        seq_len: int = 100,
        embedding_classes: int = 131,
        n_blocks: int = 2,
        n_head: int = 6,
        n_cycles: int = 1,
        res_dropout=0.1,
        att_dropout=0.0,
        n_classes: int = 2,
        learning_rate: float = 1e-3,
        class_h_bias: bool = False,
    ):
        super().__init__(
            dataset_name=dataset_name,
            d_model=d_model,
            embedding_classes=embedding_classes,
            seq_len=seq_len,
            n_blocks=n_blocks,
            n_head=n_head,
            n_cycles=n_cycles,
            res_dropout=res_dropout,
            att_dropout=att_dropout,
            n_classes=n_classes,
            learning_rate=learning_rate,
            class_h_bias=class_h_bias,
        )
        self.use_latent_input = True
        self.task = "generate"
        self.input_size = seq_len
        self.num_latent_tokens = embedding_classes

        self.embedding = LatentEmbedding(
            input_size=embedding_classes, d_model=d_model, seq_len=seq_len
        )

        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(res_dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            d_model=d_model,
                            seq_len=seq_len,
                            n_head=n_head,
                            res_dropout=res_dropout,
                            att_dropout=att_dropout,
                        )
                        for _ in range(n_blocks)
                    ]
                ),
                ln_f=nn.LayerNorm(d_model),
            )
        )
        self.lm_head = nn.Linear(d_model, embedding_classes, bias=False)

        class_head_module_dict = dict(
            linear_1=nn.Linear(d_model, 1, bias=class_h_bias),
            activation=nn.GELU(),
            linear_2=nn.Linear(seq_len, n_classes, bias=class_h_bias),
        )
        self.class_head = nn.ModuleDict(class_head_module_dict)
        # initialize weights
        self.apply(self._init_weights)
        # self._post_init_fixup()
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))

        self.log_num_params()

    def log_num_params(self) -> None:
        n_params = sum(p.numel() for p in self.transformer.parameters())
        logging.info(
            f"Transformer Blocks number of parameters: {(n_params / 1e6):.4f}M"
        )

    def forward(self, x: torch.Tensor, generate: bool = False) -> torch.Tensor:
        """
        Forward pass of the transformer decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            generate (bool): Whether to generate a sequence or perform classification

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, n_classes)
        """
        x = self.embedding(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if generate:
            logits = self.lm_head(x)
        else:
            x = self.class_head.linear_1(x)
            x = self.class_head.activation(x.squeeze(-1))
            logits = self.class_head.linear_2(x)
        return logits

    def forward_layers(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.class_head.linear_1(x)
        x = self.class_head.activation(x.squeeze(-1))
        logits = self.class_head.linear_2(x)
        return logits

        
    def step_task_gen(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and compute the loss for the generation task.

        Args:
            batch (tuple): A tuple containing the input data and labels.

        Returns:
            tuple: A tuple containing the loss, logits, and labels.
        """
        x, _, y = batch
        logits = self(x, generate=True)
        loss = self.loss_cross_entropy(logits, y, ignore_index=-1)
        return loss, logits, y

    def step_task_class(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and compute the loss for the classification task.

        Args:
            batch (tuple): A tuple containing the input data and labels.

        Returns:
            tuple: A tuple containing the loss, logits, and labels.
        """
        x, y, _ = batch
        logits = self(x, generate=False)
        loss = self.loss_cross_entropy(logits, y)
        return loss, logits, y

    def generate(self, x, do_sample=False, top_k=None) -> torch.Tensor:
        """
        Generate a sequence using the transformer decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
            do_sample (bool): Whether to sample from the distribution
            top_k (int): Top-k sampling parameter

        Returns:
            torch.Tensor: Generated sequence of shape (batch_size, sequence_length)
        """
        with torch.no_grad():
            for _ in range(self.seq_len):

                x_cond = x if x.size(1) <= self.seq_len else x[:, -self.seq_len :]
                # print(f"{x_cond.shape=} - {x.shape=}")
                logits = self(x_cond)

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                probs = probs[:, -1]
                if do_sample:
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    _, idx_next = torch.topk(probs, k=1, dim=-1)
                x = torch.cat([x, idx_next], dim=-1)
        return x

    def feature_list(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Performs a forward pass through the classification path and returns 
        the final output and a list of intermediate activations.

        Intermediate activations are collected after each Transformer block 
        and after the classification head's activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: A tuple containing:
                - The final output tensor (logits) of the classification head.
                - A list of tensors representing intermediate activations.
        """
        out_list: list[torch.Tensor] = []
        for block in self.transformer.h:
            x = block(x)
            out_list.append(x)  # Activation after each block
        x = self.transformer.ln_f(x)
        x = self.class_head.linear_1(x)
        x = self.class_head.activation(x.squeeze(-1))
        out_list.append(x)  # Activation after classification head activation
        logits = self.class_head.linear_2(x)
        return logits, out_list

    def intermediate_forward(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        """
        Performs a forward pass through the classification path up to a 
        specified intermediate layer index.

        The layers are indexed as follows:
        - 0 to n_blocks-1: Output of each Transformer block.
        - n_blocks: Output after the classification head's activation.

        Args:
            x (torch.Tensor): Input tensor.
            layer_index (int): The index of the intermediate layer whose 
                               output is desired.

        Returns:
            torch.Tensor: The output tensor of the specified intermediate layer.

        Raises:
            ValueError: If `layer_index` is out of bounds.
        """
        activation_index = -1
        max_index = self.n_blocks 
        for block_idx, block in enumerate(self.transformer.h):
            x = block(x)
            activation_index += 1
            if activation_index == layer_index:
                return x
        
        x = self.transformer.ln_f(x)
        x = self.class_head.linear_1(x)
        x = self.class_head.activation(x.squeeze(-1))
        activation_index += 1
        if activation_index == layer_index:
            return x

        raise ValueError(
            f"layer_index {layer_index} is too large. Maximum index is "
            f"{max_index}."
        )

   
