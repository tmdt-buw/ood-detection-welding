import numpy as np
import torch
from torch.utils.data import Dataset


class ReconDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return torch.tensor(self.ds[idx], dtype=torch.float32)


class ClassificationDataset(Dataset):
    def __init__(self, ds, labels):
        self.ds = ds
        self.labels = labels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return torch.tensor(self.ds[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class MyLatentAutoregressiveDataset(Dataset):
    def __init__(self, data: np.ndarray, y: np.ndarray | None = None, prob_missing: float = 0.0):
        """
        Dataset for Autoregressive model to predict the latent space 
        To achive this we shift the data to the right and add a start token and an end token
        -> [start,2,3,4] predicts [2,3,4,end]

        Tokens:
        - start token: max_token + 1
        - end token: max_token + 2
        - unknown token: max_token + 3
        
        Args:
            data (np.array): Input and target data
            y (Optional[np.array], optional): Not needed here but needed for compatibility with other dataloaders. Defaults to None.
            prob_missing (float, optional): Probability of missing tokens. Defaults to 0.0.
        """
        max_token = int(np.max(data))
        self.start_token = max_token + 1
        self.end_token = max_token + 2
        start_vec = np.full((len(data),), fill_value=self.start_token)
        end_vec = np.full((len(data),), fill_value=self.end_token)
        self.unknown_token = max_token + 3
        self.num_classes = max_token + 4
        self.prob_missing = prob_missing

        # shift right with start token
        self.data = np.concatenate([start_vec[:, None], data], axis=1)
        # add end token
        self.data_shifted = np.concatenate([data, end_vec[:, None]], axis=1)

        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.data_shifted[idx], dtype=torch.long)

        # add missing tokens if prob_missing > random value
        if self.prob_missing > np.random.rand():
            rand_idx = np.random.randint(1, len(x) - 1)
            x[rand_idx] = self.unknown_token

        if self.labels is not None:
            cond = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            cond = torch.zeros((1,), dtype=torch.long)
        return x, cond, y


class MyLatentClassificationDataset(Dataset):
    """
    Dataset class for classification task

    Args:
        data (np.ndarray): Input data
        y (np.ndarray | None): Optional labels for conditional generation
        prob_missing (float): Probability of masking tokens during training
    """

    def __init__(
        self, data: np.ndarray, y: np.ndarray, prob_missing: float = 0.0
    ):
        """
        Dataset for classification task

        Args:
            data (np.array): Input and target data
            y (Optional[np.array], optional): Not needed here but needed for compatibility with other dataloaders. Defaults to None.
            prob_missing (float, optional): Probability of missing tokens. Defaults to 0.0.
        """
        max_token = int(np.max(data))
        self.start_token = max_token + 1
        self.end_token = max_token + 2
        start_vec = np.full((len(data),), fill_value=self.start_token)
        self.unknown_token = max_token + 3
        self.num_classes = max_token + 4
        self.prob_missing = prob_missing
        # shift right with start token
        self.data = np.concatenate([start_vec[:, None], data], axis=1)

        self.labels = y

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[idx], dtype=torch.long)

        # add missing tokens if prob_missing > random value
        if self.prob_missing > np.random.rand():
            rand_idx = np.random.randint(1, len(x) - 1)
            x[rand_idx] = self.unknown_token

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, label