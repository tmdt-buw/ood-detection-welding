from torch import nn, Tensor
from model.classification_base import ClassificationLightningModule

activation_classes = (nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.GELU, nn.Tanh)


class MLP(ClassificationLightningModule):
    """
    MLP model for classification.

    Inherits from ClassificationLightningModule to provide a standard
    interface for training and evaluation.

    Attributes:
        use_latent_input (bool): If True, use latent token embeddings as input.
        num_latent_tokens (int): Number of latent tokens for embedding layer.
        embedding (nn.Embedding): Embedding layer for latent tokens.
        layers (nn.ModuleList): List of layers in the MLP.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        in_dim: int,
        hidden_sizes: int,
        annealing_step: int,
        annealing_start: float,
        n_hidden_layers: int = 4,
        dropout_p: float = 0.1,
        learning_rate: float = 1e-3,
        model_id: str = "",
        use_latent_input: bool = False,
        use_edl_loss: bool = False,
        num_latent_tokens: int = 0,
        use_layer_norm: bool = False,
    ) -> None:
        """
        Initializes the MLP model.

        Args:
            input_size: Size of the input features.
            output_size: Number of output classes.
            in_dim: Dimension of the input features (used if not using latent).
            hidden_sizes: Size of the hidden layers.
            annealing_step: Step for EDL annealing.
            annealing_start: Starting value for EDL annealing.
            n_hidden_layers: Number of hidden layers. Defaults to 4.
            dropout_p: Dropout probability. Defaults to 0.1.
            learning_rate: Learning rate for the optimizer. Defaults to 1e-3.
            model_id: Identifier for the model instance. Defaults to "".
            use_latent_input: Whether to use latent token embeddings as input.
                                Defaults to False.
            use_edl_loss: Whether to use Evidential Deep Learning loss.
                           Defaults to False.
            num_latent_tokens: Number of latent tokens if use_latent_input
                               is True. Defaults to 0.
            use_layer_norm: Whether to use layer normalization. Defaults to False.
        """
        self.use_latent_input = use_latent_input
        self.num_latent_tokens = num_latent_tokens
        super().__init__(
            input_size=input_size,
            num_classes=output_size,
            in_dim=in_dim,
            d_model=hidden_sizes,
            annealing_step=annealing_step,
            annealing_start=annealing_start,
            n_hidden_layers=n_hidden_layers,
            dropout_p=dropout_p,
            learning_rate=learning_rate,
            model_id=model_id,
            use_edl_loss=use_edl_loss,
            use_layer_norm=use_layer_norm,
        )

        self.embedding = nn.Embedding(num_latent_tokens, hidden_sizes)

        layers = nn.ModuleList(
            [
                nn.Linear(
                     (
                        input_size * hidden_sizes
                        if use_latent_input
                        else input_size * in_dim
                    ), hidden_sizes
                ),
                nn.LayerNorm(hidden_sizes),
                nn.GELU(),
            ]
        )

        for i in range(n_hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_sizes, hidden_sizes),
                    nn.LayerNorm(hidden_sizes),
                    nn.GELU(),
                ]
            )

        layers.extend(
            [
                nn.Dropout(p=dropout_p),
                nn.Linear(hidden_sizes, output_size),
            ]
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the MLP model.

        Args:
            x (Tensor): Input tensor. Expected shape depends on
                `use_latent_input`. If True, expects LongTensor indices for
                embedding lookup. Otherwise, expects FloatTensor features.

        Returns:
            Tensor: Output tensor representing class logits or evidence
                (if `use_edl_loss` is True).
        """
        x = x.reshape(x.shape[0], -1)
        if self.use_latent_input:
            x = self.embedding(x).reshape(x.shape[0], -1)
        x = self.forward_layers(x)
        return x

    def forward_layers(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers except the optional embedding layer.

        This method is useful for operations requiring gradients through the
        main MLP structure without involving the embedding layer.

        Args:
            x (Tensor): Input tensor after potential embedding lookup and
                reshaping.

        Returns:
            Tensor: Output tensor after passing through all hidden and output
                layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def feature_list(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """
        Performs a forward pass and returns the final output and a list of
        intermediate activations.

        The intermediate activations are collected after each activation layer
        (LeakyReLU, ReLU, Sigmoid).

        Args:
            x (Tensor): Input tensor.

        Returns:
            tuple[Tensor, list[Tensor]]: A tuple containing:
                - The final output tensor of the network.
                - A list of tensors representing the activations after each
                  activation layer.
        """
        out_list: list[Tensor] = []
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, activation_classes):
                out_list.append(x)

        return x, out_list

    def intermediate_forward(self, x: Tensor, layer_index: int) -> Tensor:
        """
        Performs a forward pass up to a specified activation layer index.

        Args:
            x (Tensor): Input tensor.
            layer_index (int): The index of the activation layer whose output
                is desired. Indexing starts from 0 for the first activation
                layer encountered.

        Returns:
            Tensor: The output tensor of the specified activation layer.

        Raises:
            ValueError: If `layer_index` is out of bounds (greater than or
                equal to the number of activation layers).
        """
        x = x.reshape(x.shape[0], -1)
        activation_index = -1

        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, activation_classes):
                activation_index += 1
                if activation_index == layer_index:
                    return x

        raise ValueError(
            f"layer_index {layer_index} is too large. Maximum "
            f"{activation_index}"
        )
