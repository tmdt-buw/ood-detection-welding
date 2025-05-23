import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class LearningWithoutForgetting:
    """
    Implementation of the Learning without Forgetting (LwF) algorithm.

    LwF is a continual learning method that uses knowledge distillation to prevent
    catastrophic forgetting when training a model on new tasks without access to data
    from previously learned tasks.

    Reference: Li, Z., & Hoiem, D. (2016). Learning without Forgetting.
    arXiv:1606.09282
    """

    def __init__(
        self,
        model: nn.Module,
        temperature: float = 2.0,
        lambda_old: float = 1.0,
        lambda_new: float = 1.0,
    ):
        """
        Initialize the LwF algorithm.

        Args:
            model: The neural network model to apply LwF to
            temperature: Temperature parameter for softening probability distributions
                         in knowledge distillation (higher values produce softer distributions)
            lambda_old: Weight for the old task distillation loss
            lambda_new: Weight for the new task loss
        """
        self.model = model
        self.temperature = temperature
        self.lambda_old = lambda_old
        self.lambda_new = lambda_new
        self.old_task_logits: dict[str, dict[str, list[torch.Tensor]]] = {}
        self.logger = logging.getLogger(__name__)

    def compute_distillation_loss(
        self, current_logits: torch.Tensor, target_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the knowledge distillation loss between current and target logits.

        Args:
            current_logits: Current model outputs for the old task
            target_logits: Recorded model outputs from before learning the new task

        Returns:
            Knowledge distillation loss
        """
        # Apply temperature scaling to soften the probability distributions
        current_logits_T = current_logits / self.temperature
        target_logits_T = target_logits / self.temperature

        # Convert logits to probabilities (soft targets)
        current_probs = F.softmax(current_logits_T, dim=1)
        target_probs = F.softmax(target_logits_T, dim=1)

        target_probs = target_probs.to(current_probs.device)

        # Compute KL divergence loss
        kd_loss = F.kl_div(
            F.log_softmax(current_logits_T, dim=1), target_probs, reduction="batchmean"
        ) * (self.temperature**2)

        return kd_loss

    def record_old_task_outputs(
        self,
        task_id: str,
        data_loader: torch.utils.data.DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Record the current model's outputs on the new task data before training
        on the new task. These outputs represent the knowledge of old tasks.

        Args:
            task_id: Identifier for the old task
            data_loader: DataLoader containing the new task's data
            device: Device to run inference on
        """
        self.logger.info(f"Recording old task outputs for task {task_id}")
        self.model.eval()
        all_logits = []
        all_inputs = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple) or isinstance(batch, list):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                # Store inputs for matching with model outputs later
                all_inputs.append(x.cpu())

                # Forward pass to get logits
                logits = self.model(x, generate=False)
                all_logits.append(logits.cpu())

        # Store the logits and inputs as lists
        self.old_task_logits[task_id] = {"inputs": all_inputs, "logits": all_logits}

        self.logger.info(
            f"Recorded outputs for {sum(len(batch) for batch in all_inputs)} samples for task {task_id}"
        )

    def compute_combined_loss(
        self, new_task_loss: torch.Tensor, old_task_id: str, current_input: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
        Compute the combined loss with knowledge distillation for LwF.

        Args:
            new_task_loss: Standard loss for the new task
            old_task_id: Identifier for the old task
            current_input: Current input data batch
            batch_idx: Index of the batch in the data loader
        Returns:
            Combined loss for Learning without Forgetting
        """
        if old_task_id not in self.old_task_logits:
            self.logger.warning(f"No recorded outputs found for task {old_task_id}")
            return new_task_loss

        old_task_data = self.old_task_logits[old_task_id]

        # Get the current model's output on the same data
        current_logits = self.model(current_input, generate=False)

        # Compute the knowledge distillation loss
        distillation_loss = self.compute_distillation_loss(
            current_logits, old_task_data["logits"][batch_idx]
        )

        # Combine the losses
        total_loss = (
            self.lambda_new * new_task_loss + self.lambda_old * distillation_loss
        )

        return total_loss

