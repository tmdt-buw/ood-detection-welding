import logging as log
from typing import Literal
from torch.utils.data import DataLoader
import numpy as np
import lightning.pytorch as pl

from data_loader.utils import train_val_test_split, create_sequence_ds, MyScaler
from data_loader.datasets import ReconDataset, ClassificationDataset


class WeldingDataModule(pl.LightningDataModule):
    """
    LightningDataModule subclass for handling data loading and preprocessing.

    Args:
        ds (np.ndarray): The input data array.
        labels (np.ndarray): The labels array corresponding to the input data.
        val_split_idx (np.ndarray | None, optional): The indices to use for validation split. Defaults to None.
        test_split_idx (np.ndarray | None, optional): The indices to use for test split. Defaults to None.
        batch_size (int, optional): The batch size for data loading. Defaults to 256.
        shuffle_train (bool, optional): Whether to shuffle the training data. Defaults to True.
        n_cycles (int, optional): The number of cycles for the sequence dataset. Defaults to 1.
        ds_type (str, optional): The type of dataset to create. Defaults to "reconstruction".
    """

    def __init__(
        self,
        ds: np.ndarray,
        labels: np.ndarray,
        exp_ids: np.ndarray,
        val_split_idx: np.ndarray | None = None,
        test_split_idx: np.ndarray | None = None,
        batch_size: int = 256,
        shuffle_train: bool = True,
        n_cycles: int = 1,
        ds_type: Literal["reconstruction", "classification"] = "reconstruction"
    ) -> None:
        super().__init__()
        self.ds = ds
        self.labels = labels
        self.exp_ids = exp_ids
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.train_ds: ClassificationDataset | ReconDataset
        self.val_ds: ClassificationDataset | ReconDataset
        self.test_ds: ClassificationDataset | ReconDataset
        self.val_idx: np.ndarray | None = val_split_idx
        self.test_idx: np.ndarray | None = test_split_idx
        self.scaler = MyScaler()
        self.n_cycles = n_cycles
        self.ds_type = ds_type

    def scale_data(self, x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray):
        """
        Scale the input data using a scaler object.

        Args:
            x_train (np.ndarray): The training data array.
            x_val (np.ndarray): The validation data array.
            x_test (np.ndarray): The test data array.

        Returns:
            np.ndarray: The scaled training data array.
            np.ndarray: The scaled validation data array.
            np.ndarray: The scaled test data array.
        """
        self.scaler.fit(x_train)
        x_train = self.scaler.transform(x_train)
        x_val = self.scaler.transform(x_val)
        x_test = self.scaler.transform(x_test)
        return x_train, x_val, x_test

    def split_train_val_test_by_index(
        self, train_x: np.ndarray, train_y: np.ndarray, seq_len: int = 1
    ):
        """
        Split the training data into training, validation, and test sets based on provided indices.

        Args:
            train_x (np.ndarray): The training data array.
            train_y (np.ndarray): The training labels array.
            seq_len (int, optional): The sequence length for creating sequence datasets. Defaults to 1.

        Returns:
            np.ndarray: The training data array.
            np.ndarray: The validation data array.
            np.ndarray: The test data array.
            np.ndarray: The training labels array.
            np.ndarray: The validation labels array.
            np.ndarray: The test labels array.
        """
        num_samples = train_x.shape[0]
        val_idx = self.val_idx
        test_idx = self.test_idx
        if seq_len > 1:
            val_idx = val_idx[val_idx < num_samples - seq_len]
            test_idx = test_idx[test_idx < num_samples - seq_len]

        val_test_indices = np.concatenate([val_idx, test_idx])

        train_idx = ~np.isin(np.arange(num_samples), val_test_indices)
        x_train = train_x[train_idx]
        x_val = train_x[val_idx]
        x_test = train_x[test_idx]
        y_train = train_y[train_idx]
        y_val = train_y[val_idx]
        y_test = train_y[test_idx]
        return x_train, x_val, x_test, y_train, y_val, y_test

    def setup(
        self,
        stage=None,
    ):
        """
        Setup the data module by preprocessing the data and creating the appropriate datasets.

        Args:
            stage (None, optional): The current stage of training. Defaults to None.
            ds_type (str, optional): The type of dataset to create. Defaults to "reconstruction".
            seq_len (int, optional): The sequence length for creating sequence datasets. Defaults to 1.
        """
        if self.n_cycles > 1:
            ds, labels = create_sequence_ds(self.ds, self.labels, self.n_cycles)
            log.info(
                f"Creating sequence dataset {self.n_cycles} | new dataset shape: {ds.shape}"
            )
        else:
            ds, labels = self.ds, self.labels

        if self.val_idx is not None and self.test_idx is not None:
            log.info("Using provided val and test indices")
            x_train, x_val, x_test, y_train, y_val, y_test = (
                self.split_train_val_test_by_index(
                    train_x=ds, train_y=labels, seq_len=self.n_cycles
                )
            )
        else:
            x_train, x_val, x_test, y_train, y_val, y_test, val_idx, test_idx = (
                train_val_test_split(
                    train_x=ds,
                    train_y=labels,
                    val_size=0.1,
                    exp_ids=self.exp_ids,
                    ood_experiments=[3, 4],
                )
            )
            self.val_idx = val_idx
            self.test_idx = test_idx
        log.info(
            f"Train shape: {x_train.shape} Val shape: {x_val.shape} Test shape: {x_test.shape}"
        )
        log.info(
            f"Train labels shape: {y_train.shape} Val labels shape: {y_val.shape} Test labels shape: {y_test.shape}"
        )
        x_train, x_val, x_test = self.scale_data(
            x_train=x_train, x_val=x_val, x_test=x_test
        )

        if self.ds_type == "reconstruction":
            self.train_ds = ReconDataset(x_train)
            self.val_ds = ReconDataset(x_val)
            self.test_ds = ReconDataset(x_test)
        elif self.ds_type == "classification":
            x_train, y_train = self.filter_out_not_labbeld(x_train, y_train)
            x_val, y_val = self.filter_out_not_labbeld(x_val, y_val)
            x_test, y_test = self.filter_out_not_labbeld(x_test, y_test)

            self.train_ds = ClassificationDataset(x_train, y_train)
            self.val_ds = ClassificationDataset(x_val, y_val)
            self.test_ds = ClassificationDataset(x_test, y_test)
        else:
            raise ValueError("ds_type must be 'reconstruction' or 'classification'")

    @staticmethod
    def filter_out_not_labbeld(ds: np.ndarray, labels: np.ndarray):
        """
        Filter out samples with labels equal to -1.

        Args:
            ds (np.ndarray): The input data array.
            labels (np.ndarray): The labels array.

        Returns:
            np.ndarray: The filtered data array.
            np.ndarray: The filtered labels array.
        """
        return ds[labels != -1], labels[labels != -1]

    def train_dataloader(self):
        """
        Return the data loader for training data.

        Returns:
            torch.utils.data.DataLoader: The data loader for training data.
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=8,
        )

    def val_dataloader(self):
        """
        Return the data loader for validation data.

        Returns:
            torch.utils.data.DataLoader: The data loader for validation data.
        """
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def test_dataloader(self):
        """
        Return the data loader for test data.

        Returns:
            torch.utils.data.DataLoader: The data loader for test data.
        """
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)


class SimpleDataModule(pl.LightningDataModule):

    def __init__(self, train_ds, val_ds, test_ds, batch_size: int = 512) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.train_ds: ClassificationDataset | ReconDataset = train_ds
        self.val_ds: ClassificationDataset | ReconDataset = val_ds
        self.test_ds: ClassificationDataset | ReconDataset = test_ds

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
