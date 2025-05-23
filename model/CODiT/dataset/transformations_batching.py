import numpy as np
import random
import torch
from scipy.ndimage import maximum_filter1d, minimum_filter1d

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import random
from data_loader.datasets import ClassificationDataset


class GaitBatching(Dataset):
    """Simplified GAIT dataloader function for transformations"""

    def __init__(
        self,
        data: ClassificationDataset,
        win_len: int = 200,
        transformation_list: list[str] = [
            "low_pass",
            "high_pass",
            "dilation",
            "erosion",
            "identity",
        ],
    ):
        """
        Args:
            data (ClassificationDataset): Time series data of shape (time_steps, features, channels)
            win_len (int): Number of frames in each window
            transformation_list (list): List of transformations to apply on data windows
        """
        self.data = data
        self.win_len = win_len
        self.transformation_list = transformation_list
        self.num_classes = len(self.transformation_list)

    def __len__(self) -> int:
        return len(self.data) - self.win_len + 1

    def transform_win(self, win: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(win).unsqueeze(
            0
        )  # Shape becomes (1, win_len, features)

    def apply_filter_on_2D_data(
        self, input_data: np.ndarray, filter_coeffs: np.ndarray
    ) -> np.ndarray:
        return np.apply_along_axis(
            lambda m: np.convolve(m, filter_coeffs, mode="valid"),
            axis=0,
            arr=input_data,
        )

    def apply_dilation(
        self, input_data: np.ndarray, filter_size: int = 3
    ) -> np.ndarray:
        return np.apply_along_axis(
            lambda m: maximum_filter1d(m, size=filter_size), axis=0, arr=input_data
        )

    def apply_erosion(self, input_data: np.ndarray, filter_size: int = 3) -> np.ndarray:
        return np.apply_along_axis(
            lambda m: minimum_filter1d(m, size=filter_size), axis=0, arr=input_data
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, np.ndarray]:
        # Extract original window
        orig_win = self.data.ds[idx]
        quality_labels = self.data.labels[idx]
        trans_win = np.copy(orig_win)

        # Select a random transformation
        transform_id = random.randint(0, self.num_classes - 1)
        transform_type = self.transformation_list[transform_id]

        # Apply transformations
        if transform_type == "low_pass":
            trans_win = self.apply_filter_on_2D_data(trans_win, np.array([1 / 3, 1 / 3, 1 / 3]))
        elif transform_type == "high_pass":
            trans_win = self.apply_filter_on_2D_data(trans_win, np.array([-1 / 2, 0, 1 / 2]))
        elif transform_type == "dilation":
            trans_win = self.apply_dilation(trans_win, filter_size=3)
        elif transform_type == "erosion":
            trans_win = self.apply_erosion(trans_win, filter_size=3)

        # converting to tensors and new dim = 1 (no. of channels for conv) X win_len X 12
        trans_win = self.transform_win(trans_win)

        orig_win = self.transform_win(orig_win)
        orig_win = orig_win[:, 1:-1, :]

        if (
            self.transformation_list[transform_id] == "identity"
            or self.transformation_list[transform_id] == "dilation"
            or self.transformation_list[transform_id] == "erosion"
        ):
            trans_win = trans_win[:, 1:-1, :]

        return orig_win, trans_win, transform_id, quality_labels
