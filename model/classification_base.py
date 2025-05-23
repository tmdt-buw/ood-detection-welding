from abc import abstractmethod
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
import lightning.pytorch as pl
from model.edl_approach.edl_loss import edl_log_loss
from lightning.pytorch.loggers.mlflow import MLFlowLogger


class ClassificationLightningModule(pl.LightningModule):
    """
    Base class for classification models using PyTorch Lightning.

    Implements common classification functionality including training loop,
    metrics tracking (accuracy, F1 score), and optimization setup.

    Args:
        input_size (int): Size of input features
        num_classes (int): Number of output classes
        in_dim (int): Input dimension
        d_model (int): Size of hidden layers
        annealing_step (float): Annealing step for EDL loss
        annealing_start (float): Annealing start epoch for EDL loss
        n_hidden_layers (int, optional): Number of hidden layers. Defaults to 4
        dropout_p (float, optional): Dropout probability. Defaults to 0.1
        learning_rate (float, optional): Learning rate. Defaults to 1e-3
        model_id (str, optional): Model identifier string. Defaults to "".
        use_edl_loss (bool, optional): Whether to use EDL loss. Defaults to True.
        use_layer_norm (bool, optional): Whether to use layer normalization. Defaults to False.
        use_log_vacuity_histogram (bool, optional): Whether to log vacuity histograms. Defaults to False.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        in_dim: int,
        d_model: int,
        annealing_step: float,
        annealing_start: float,
        n_hidden_layers: int = 4,
        dropout_p: float = 0.1,
        learning_rate: float = 1e-3,
        model_id: str = "",
        use_edl_loss: bool = True,
        use_layer_norm: bool = False,
        use_log_vacuity_histogram: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.d_model = d_model
        self.annealing_step = annealing_step
        self.annealing_start = annealing_start
        self.n_hidden_layers = n_hidden_layers
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        self.model_id = f"{model_id}/" if model_id != "" else ""
        self.use_edl_loss = use_edl_loss
        self.use_log_vacuity_histogram = use_log_vacuity_histogram
        self.use_layer_norm = use_layer_norm
        self.best_val_score = 0
        task = "binary" if num_classes == 2 else "multiclass"

        self.val_losses = []
        self.train_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_accuracy = Accuracy(task=task, num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.vacuity_values_train = []
        self.vacuity_values_val = []
        self.vacuity_values_test = []

        self.save_hyperparameters()

    @abstractmethod
    def forward(
        self, x: torch.Tensor   
    ) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Reconstruction loss, data reconstruction, perplexity
        """
        raise NotImplementedError

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Loss function. Returns loss and optionally vacuity if EDL is used.

        Args:
            logits (torch.Tensor): Logits
            labels (torch.Tensor): Labels

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Loss value and optional vacuity tensor.
        """
        loss, vacuity = None, None

        if self.use_edl_loss:
            loss, vacuity = edl_log_loss(
                logits,
                labels,
                self.trainer.current_epoch,
                self.num_classes,
                self.annealing_step,
                self.annealing_start,
                device=logits.device,
            )
        else:
            loss = F.cross_entropy(logits, labels)
        return loss, vacuity

    def _get_preds(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        logits = self(x)
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        loss, vacuity = self.loss(logits, y)
        return preds, loss, vacuity, logits

    def log_vacuity_histogram(self, vacuity, step, epoch, type):
        flattened_tensor = vacuity.flatten()

        if flattened_tensor.device.type == "cuda":
            flattened_tensor = flattened_tensor.cpu()

        saving_path = Path(f"artifacts/{type}/epoch_{epoch}/")
        saving_path.mkdir(parents=True, exist_ok=True)
        histogram_path = saving_path / f"batch_{step}.png"

        plt.hist(flattened_tensor.detach().numpy(), bins=30, color="blue", alpha=0.7)
        plt.title("Vacuity Distribution")
        plt.xlabel("Vacuity")
        plt.xlim(0, 1)
        plt.grid(True)
        plt.savefig(histogram_path)
        plt.close()

        if isinstance(self.logger, MLFlowLogger):
            self.logger.experiment.log_artifact(
                run_id=self.logger.run_id,
                local_path=str(histogram_path),
                artifact_path=f"{type}/epoch_{epoch}",
            )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        PyTorch Lightning calls this inside the training loop
        """
        x, y = batch
        preds, loss, vacuity, logits = self._get_preds(x, y)
        acc = self.train_accuracy(preds, y)
        f1score = self.train_f1(preds, y)

        self.log(f"{self.model_id}train/loss", loss.item(), on_step=True, on_epoch=False)
        self.log(f"{self.model_id}train/acc", acc.item(), on_step=True, on_epoch=False)
        self.log(f"{self.model_id}train/f1_score", f1score.item(), on_step=True, on_epoch=False, prog_bar=True)

        if self.use_edl_loss and vacuity is not None:
            vacuity_filtered = vacuity[~torch.isnan(vacuity)]
            if vacuity_filtered.numel() > 0:
                self.vacuity_values_train.append(vacuity_filtered)
                average_vacuity_batch = torch.mean(vacuity_filtered)
                self.log(f"{self.model_id}train/average_vacuity_batch", average_vacuity_batch.item(), on_step=True, on_epoch=False)

            if batch_idx + 1 == self.trainer.num_training_batches and self.vacuity_values_train:
                summarized_vacuity = torch.cat(self.vacuity_values_train, dim=0)
                if self.use_log_vacuity_histogram:
                    self.log_vacuity_histogram(
                        summarized_vacuity,
                        "summarized",
                        self.trainer.current_epoch,
                        "train",
                    )
                average_vacuity_summarized = torch.mean(summarized_vacuity)
                self.log(
                    f"{self.model_id}train/average_vacuity_summarized",
                    average_vacuity_summarized.item(),
                    on_step=False,
                    on_epoch=True,
                )
                self.vacuity_values_train = []

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        PyTorch Lightning calls this inside the validation loop
        """
        x, y = batch
        preds, loss, vacuity, logits = self._get_preds(x, y)
        self.val_accuracy(preds, y)
        self.val_f1(preds, y)
        self.val_losses.append(loss.item())

        if self.use_edl_loss and vacuity is not None:
            vacuity_filtered = vacuity[~torch.isnan(vacuity)]
            if vacuity_filtered.numel() > 0:
                 self.vacuity_values_val.append(vacuity_filtered)

        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        PyTorch Lightning calls this inside the test loop
        """
        x, y = batch
        preds, loss, vacuity, logits = self._get_preds(x, y)
        self.test_accuracy(preds, y)
        self.test_f1(preds, y)

        if self.use_edl_loss and vacuity is not None:
            vacuity_filtered = vacuity[~torch.isnan(vacuity)]
            if vacuity_filtered.numel() > 0:
                 self.vacuity_values_test.append(vacuity_filtered)

        return loss

    def on_validation_epoch_start(self):
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.val_losses = []
        self.vacuity_values_val = []
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        val_acc = self.val_accuracy.compute()
        val_f1 = self.val_f1.compute()
        val_loss = np.mean(self.val_losses) if self.val_losses else torch.tensor(0.0)
        self.log(f"{self.model_id}val/f1_score", val_f1, sync_dist=True, prog_bar=True)
        self.log(f"{self.model_id}val/acc", val_acc, sync_dist=True, prog_bar=True)
        self.log(f"{self.model_id}val/loss", val_loss, sync_dist=True, prog_bar=True)

        if self.use_edl_loss and self.vacuity_values_val:
            summarized_vacuity = torch.cat(self.vacuity_values_val, dim=0)
            if self.use_log_vacuity_histogram:
                self.log_vacuity_histogram(
                    summarized_vacuity, "summarized", self.trainer.current_epoch, "val"
                )
            average_vacuity_summarized = torch.mean(summarized_vacuity)
            self.log(f"{self.model_id}val/average_vacuity_summarized", average_vacuity_summarized, sync_dist=True)
            self.vacuity_values_val = []

        if val_f1 > self.best_val_score:
            self.best_val_score = val_f1
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.vacuity_values_test = []
        return super().on_test_epoch_start()

    def on_test_epoch_end(self):
        test_acc = self.test_accuracy.compute()
        test_f1 = self.test_f1.compute()

        self.log(f"{self.model_id}test/f1_score", test_f1, sync_dist=True, prog_bar=True)
        self.log(f"{self.model_id}test/acc", test_acc, sync_dist=True, prog_bar=True)

        if self.use_edl_loss and self.vacuity_values_test:
            summarized_vacuity = torch.cat(self.vacuity_values_test, dim=0)
            if self.use_log_vacuity_histogram:
                self.log_vacuity_histogram(
                    summarized_vacuity, "summarized", self.trainer.current_epoch, "test"
                )
            average_vacuity_summarized = torch.mean(summarized_vacuity)
            self.log(f"{self.model_id}test/average_vacuity_summarized", average_vacuity_summarized, sync_dist=True)
            self.vacuity_values_test = []

        return super().on_test_epoch_end()

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.RAdam(self.parameters(), lr=self.learning_rate)
        return optimizer
