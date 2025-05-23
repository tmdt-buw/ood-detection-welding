"""
Implementation of Experience Replay for continual learning.

This module provides a memory buffer that stores samples from previous tasks
and allows them to be replayed during training on new tasks to mitigate
catastrophic forgetting.
"""

import logging
import random
from collections import deque
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class ReplayMemory:
    """
    Maintains a buffer of samples from previous tasks for experience replay.

    This class implements a memory buffer with reservoir sampling to store
    a representative set of examples from previous tasks. These examples can
    be replayed during training on new tasks to mitigate catastrophic forgetting.
    """

    def __init__(
        self,
        capacity: int,
        sample_selection: str = "random",
        task_balanced: bool = True,
        device: str = "cpu",
        is_gen_dataset: bool = False,
    ):
        """
        Initialize the replay memory buffer.

        Args:
            capacity: Maximum number of samples to store in the buffer
            sample_selection: Strategy for selecting samples ('random', 'loss', 'entropy')
            task_balanced: Whether to maintain the same number of samples per task
            device: Device to store the samples on
        """
        self.capacity = capacity
        self.sample_selection = sample_selection
        self.task_balanced = task_balanced
        self.device = device
        self.buffer: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self.per_task_capacity = {}
        self.current_size = 0
        self.is_gen_dataset = is_gen_dataset
        self.logger = logging.getLogger(__name__)

    def add_samples(
        self,
        task_id: str,
        samples: List[Tuple[torch.Tensor, torch.Tensor]],
        importance_scores: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add samples from a task to the replay buffer.

        Args:
            task_id: Identifier for the task
            samples: List of (input, target) pairs to store
            importance_scores: Optional importance scores for sample selection
        """
        if task_id not in self.buffer:
            self.buffer[task_id] = []

        # Update per-task capacity if task_balanced is True
        if self.task_balanced and len(self.buffer) > 0:
            self.per_task_capacity = {
                tid: self.capacity // len(self.buffer) for tid in self.buffer
            }

        # Determine samples to add based on selection strategy
        if importance_scores is not None and self.sample_selection in [
            "loss",
            "entropy",
        ]:
            # Select top samples based on importance scores
            num_to_add = min(
                len(samples), self.per_task_capacity.get(task_id, self.capacity)
            )
            indices = np.argsort(importance_scores)[-num_to_add:]
            selected_samples = [samples[i] for i in indices]
        else:
            # Random selection if no importance scores provided
            selected_samples = samples

        # Reservoir sampling for the current task
        self._update_buffer(task_id, selected_samples)

        # Update current size
        self.current_size = sum(len(samples) for samples in self.buffer.values())
        self.logger.info(
            f"Added samples for task {task_id}. Current buffer size: {self.current_size}"
        )

    def _update_buffer(
        self, task_id: str, new_samples: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """
        Update the buffer for a specific task using reservoir sampling.

        Args:
            task_id: Identifier for the task
            new_samples: New samples to consider adding
        """
        task_capacity = self.per_task_capacity.get(task_id, self.capacity)

        # If we have space, just add the samples
        if len(self.buffer[task_id]) + len(new_samples) <= task_capacity:
            if self.is_gen_dataset:
                self.buffer[task_id].extend(
                    [(x.to(self.device), y.to(self.device), y_gen.to(self.device)) for x, y, y_gen in new_samples]
                )
            else:
                self.buffer[task_id].extend(
                    [(x.to(self.device), y.to(self.device)) for x, y in new_samples]
                )
            return

        # If we need to perform reservoir sampling
        current_samples = self.buffer[task_id]

        # Combine current and new samples
        if self.is_gen_dataset:
            all_samples = current_samples + [
                (x.to(self.device), y.to(self.device), y_gen.to(self.device)) for x, y, y_gen in new_samples
            ]
        else:
            all_samples = current_samples + [
                (x.to(self.device), y.to(self.device)) for x, y in new_samples
            ]

        # Randomly select samples to keep
        self.buffer[task_id] = random.sample(all_samples, task_capacity)

    def get_replay_dataloader(
        self,
        batch_size: int,
        exclude_task_id: Optional[str] = None,
    ) -> Optional[DataLoader]:
        """
        Create a DataLoader containing replay samples for training.

        Args:
            batch_size: Batch size for the DataLoader
            exclude_task_id: Task ID to exclude from the replay (typically the current task)

        Returns:
            DataLoader containing replay samples, or None if buffer is empty
        """
        if self.current_size == 0:
            return None

        # Collect samples from all tasks except the excluded one
        replay_x = []
        replay_y = []
        replay_y_gen = []

        for task_id, samples in self.buffer.items():
            if task_id == exclude_task_id:
                continue

            if self.is_gen_dataset:
                for x, y, y_gen in samples:
                    replay_x.append(x)
                    replay_y.append(y)
                    replay_y_gen.append(y_gen)
            else:
                for x, y in samples:
                    replay_x.append(x)
                    replay_y.append(y)

        if not replay_x:
            return None

        # Stack tensors
        replay_x = torch.stack(replay_x)
        replay_y = torch.stack(replay_y)
        
        # Create dataset and dataloader
        if self.is_gen_dataset:
            replay_y_gen = torch.stack(replay_y_gen)
            replay_dataset = TensorDataset(replay_x, replay_y, replay_y_gen)
        else:
            replay_dataset = TensorDataset(replay_x, replay_y)
        return DataLoader(replay_dataset, batch_size=batch_size, shuffle=True)

    def get_task_samples(self, task_id: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get all stored samples for a specific task.

        Args:
            task_id: Identifier for the task

        Returns:
            List of (input, target) pairs for the specified task
        """
        return self.buffer.get(task_id, [])

    def update_importance(
        self,
        model: torch.nn.Module,
        task_id: str,
        criterion: torch.nn.Module,
    ) -> None:
        """
        Update sample importance based on current model performance.

        For loss-based or entropy-based selection strategies, this method
        recalculates importance scores based on the current model.

        Args:
            model: Current model
            task_id: Task ID to update importance for
            criterion: Loss function to use for importance calculation
        """
        if self.sample_selection not in ["loss", "entropy"]:
            return

        if task_id not in self.buffer:
            return

        model.eval()
        samples = self.buffer[task_id]
        importance_scores = []

        with torch.no_grad():
            for items in samples:
                if self.is_gen_dataset:
                    x, y, y_gen = items
                else:
                    x, y = items
                # Get model output
                output = model(x.unsqueeze(0))

                if self.sample_selection == "loss":
                    # Use loss as importance
                    loss = criterion(output, y.unsqueeze(0))
                    importance_scores.append(loss.item())
                else:  # entropy
                    # Use prediction entropy as importance
                    probs = torch.softmax(output, dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    importance_scores.append(entropy.item())

        # Resort samples based on importance
        importance_scores = np.array(importance_scores)
        sorted_indices = np.argsort(importance_scores)
        self.buffer[task_id] = [samples[i] for i in sorted_indices]

    def merge_with(self, other_memory: "ReplayMemory") -> None:
        """
        Merge this replay memory with another one.

        Args:
            other_memory: Another ReplayMemory instance to merge with
        """
        for task_id, samples in other_memory.buffer.items():
            if task_id not in self.buffer:
                self.buffer[task_id] = []

            self._update_buffer(task_id, samples)

        # Update current size
        self.current_size = sum(len(samples) for samples in self.buffer.values())

        # Update per-task capacity if task_balanced is True
        if self.task_balanced and len(self.buffer) > 0:
            self.per_task_capacity = {
                tid: self.capacity // len(self.buffer) for tid in self.buffer
            }

    def clear(self) -> None:
        """Clear all samples from the replay memory."""
        self.buffer = {}
        self.current_size = 0


class CombinedDataLoader:
    """
    DataLoader that interleaves samples from current experience and replay memory.

    This dataloader combines samples from the current experience and replay memory
    to create mixed batches that help prevent catastrophic forgetting while learning
    new patterns effectively.

    Args:
        current_dataloader: DataLoader for current experience samples
        replay_dataloader: DataLoader for replay samples (can be None)
        batch_size: Size of the combined batches
    """

    def __init__(
        self,
        current_dataloader: DataLoader,
        replay_dataloader: Optional[DataLoader] = None,
        batch_size: int = 256,
    ):
        self.current_dataloader = current_dataloader
        self.replay_dataloader = replay_dataloader
        self.batch_size = batch_size

        self.current_iterator = None
        self.replay_iterator = None

        # Calculate length based on the total number of batches
        current_len = len(current_dataloader)
        replay_len = 0 if replay_dataloader is None else len(replay_dataloader)
        self.length = current_len + replay_len

    def __iter__(self):
        self.current_iterator = iter(self.current_dataloader)
        if self.replay_dataloader is not None:
            self.replay_iterator = iter(self.replay_dataloader)

        # Flag to alternate between current and replay
        self.use_current = True
        return self

    def __next__(self):
        if self.use_current:
            try:
                batch = next(self.current_iterator)
                if self.replay_dataloader is not None:
                    self.use_current = False
                return batch
            except StopIteration:
                if self.replay_dataloader is None or self.replay_iterator is None:
                    raise StopIteration
                # If current is exhausted, switch to replay
                self.use_current = False

        # Try getting from replay
        if not self.use_current and self.replay_iterator is not None:
            try:
                batch = next(self.replay_iterator)
                self.use_current = True
                return batch
            except StopIteration:
                # If both iterators are exhausted
                raise StopIteration

        # If we get here, both iterators are exhausted
        raise StopIteration

    def __len__(self):
        return self.length


def store_samples_in_replay(
    replay_memory: ReplayMemory,
    dataloader: DataLoader,
    task_id: str,
) -> None:
    """
    Store samples from the current experience in the replay memory.

    Args:
        replay_memory: Replay memory to store samples in
        dataloader: DataLoader containing samples to store
        task_id: Identifier for the current task
    """
    samples = []

    # Collect samples and convert to CPU for storage
    for batch in dataloader:
        if len(batch) == 3:
            batch_x, batch_y, batch_y_gen = batch
            for batch_x_i, batch_y_i, batch_y_gen_i in zip(
                batch_x, batch_y, batch_y_gen
            ):
                samples.append((batch_x_i.cpu(), batch_y_i.cpu(), batch_y_gen_i.cpu()))
        else:
            batch_x, batch_y = batch
            for batch_x_i, batch_y_i in zip(batch_x, batch_y):
                samples.append((batch_x_i.cpu(), batch_y_i.cpu()))

    logging.info(f"Collecting {len(samples)} samples for task {task_id}")
    replay_memory.add_samples(task_id, samples)
