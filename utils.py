from pathlib import Path
import hashlib
import os
from argparse import ArgumentTypeError
import logging as log
import numpy as np
import torch

from mlflow_logger import CustomLightningMLFlowLogger
from data_loader.data_module import WeldingDataModule
from data_loader.laten_ds_helper import create_autoreg_ds
from data_loader.utils import (
    save_all_ds_ids,
    load_all_ds_ids,
    load_raw_data,
)
from model.vq_vae_patch_embed import VQVAEPatch
from model.transformer_decoder import MyTransformerDecoder
from model.mlp import MLP
from data_loader.data_module import SimpleDataModule
from params import DATASET_NAMES, MODEL_NAMES


def check_dataset_name(arg_value: str) -> str:
    """
    Validate that the dataset name is one of the allowed values.

    Args:
        arg_value (str): Name of dataset to validate

    Returns:
        str: Validated dataset name

    Raises:
        ArgumentTypeError: If dataset name is invalid
    """
    valid_names = tuple(DATASET_NAMES.__args__)  # Convert to tuple for joining
    if arg_value not in valid_names:
        raise ArgumentTypeError(
            f"Invalid dataset name: {arg_value}. Choose from: {', '.join(valid_names)}"
        )
    return arg_value


def check_model_name(arg_value: str) -> str:
    """
    Validate that the model name is one of the allowed values.

    Args:
        arg_value (str): Name of model to validate

    Returns:
        str: Validated model name

    Raises:
        ArgumentTypeError: If model name is invalid
    """
    valid_names = tuple(MODEL_NAMES.__args__)  # Convert to tuple for joining
    if arg_value not in valid_names:
        raise ArgumentTypeError(
            f"Invalid model name: {arg_value}. Choose from: {', '.join(valid_names)}"
        )
    return arg_value


def get_logger(use_mlflow: bool = True, experiment_name: str = "ood-welding"):
    """
    Get the appropriate logger based on configuration.

    Args:
        use_mlflow (bool): Whether to use MLflow logger

    Returns:
        Logger: MLflow logger if use_mlflow is True, else None
    """
    if use_mlflow:
        logger = CustomLightningMLFlowLogger(
            experiment_name=experiment_name, log_model=True
        )
    else:
        logger = None
    return logger


def get_model_hash(params: dict[str, any]) -> str:
    """
    Creates a unique hash based on model parameters.

    Returns:
        str: A unique hash string representing the model configuration
    """
    # Create a sorted string representation of parameters
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))

    # Use hash function to create a unique identifier
    return hashlib.md5(param_str.encode()).hexdigest()


def load_val_test_idx(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load validation and test indices from numpy files.

    Args:
        data_path (Path): Path to directory containing val_idx.npy and test_idx.npy

    Returns:
        tuple: (validation indices array, test indices array)
    """
    val_idx = np.load(f"{data_path}/val_idx.npy")
    test_idx = np.load(f"{data_path}/test_idx.npy")

    return val_idx, test_idx


def load_raw_data_experiments(data_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if isinstance(data_path, str):
        data_path = Path(data_path)

    cycles_path = data_path / "ds_1_4_data.npy"
    id_data_path = data_path / "ds_1_4_quality.npy"

    ds = np.load(cycles_path)
    id_ds = np.load(id_data_path)
    labels = id_ds[:, 3]
    exp_ids = id_ds[:, 1]
    welding_run_ids = id_ds[:, 2]

    return ds, labels, exp_ids, welding_run_ids

def load_first_stage_model(model_path: Path) -> VQVAEPatch:
    model_name = model_path.parent.stem
    if model_name == "VQ-VAE":
        model: VQVAEPatch = VQVAEPatch.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def get_laten_ds(
    vq_vae_path: Path,
    init_ds: bool = True,
    data_path: str = "data",
    n_cycles: int = 1,
    prob_unk_token: float = 0.0,
    seq_prediction_task: bool = True,
):  
    dataset_name = "Welding"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path(data_path)

    if seq_prediction_task:
        data_id_path = data_path / f"ds_latent_ids_{vq_vae_path.stem}_{n_cycles}"
    else:
        data_id_path = data_path / f"ds_latent_ids_{vq_vae_path.stem}_{n_cycles}_CO"

    vq_vae_model = load_first_stage_model(vq_vae_path)
    vq_vae_model.eval()

    if not data_id_path.exists() or init_ds:
        log.info(f"Creating new datasets for {dataset_name} with {vq_vae_path.stem} and n_cycles={n_cycles}")
        val_idx, test_idx = load_val_test_idx(data_path / dataset_name)

        ds, labels, exp_ids = load_raw_data(data_path / dataset_name)

        data_module: WeldingDataModule = WeldingDataModule(
            ds=ds,
            labels=labels,
            exp_ids=exp_ids,
            batch_size=1024,
            shuffle_train=False,
            val_split_idx=val_idx,
            test_split_idx=test_idx,
            n_cycles=n_cycles,
            ds_type="reconstruction",

        )
        recon_train_ds, recon_val_ds, recon_test_ds = create_autoreg_ds(
            vq_vae_model,
            data_module,
            task="reconstruction",
            seq_len=n_cycles,
            device=device,
            prob_unk_token=prob_unk_token,
        )

        data_module: WeldingDataModule = WeldingDataModule(
            ds=ds,
            labels=labels,
            exp_ids=exp_ids,
            batch_size=1024,
            shuffle_train=False,
            val_split_idx=val_idx,
            test_split_idx=test_idx,
            n_cycles=n_cycles,
            ds_type="classification",
        )

        class_train_ds, class_val_ds, class_test_ds = create_autoreg_ds(
            vq_vae_model,
            data_module,
            task="classification",
            seq_len=n_cycles,
            device=device,
            prob_unk_token=prob_unk_token,
            seq_prediction_task=seq_prediction_task,
        )

        save_all_ds_ids(
            recon_train_ds=recon_train_ds,
            recon_val_ds=recon_val_ds,
            recon_test_ds=recon_test_ds,
            class_train_ds=class_train_ds,
            class_val_ds=class_val_ds,
            class_test_ds=class_test_ds,
            path=data_id_path,
        )
    else:
        (
            recon_train_ds,
            recon_val_ds,
            recon_test_ds,
            class_train_ds,
            class_val_ds,
            class_test_ds,
        ) = load_all_ds_ids(path=data_id_path)

    return (
        recon_train_ds,
        recon_val_ds,
        recon_test_ds,
        class_train_ds,
        class_val_ds,
        class_test_ds,
    )


def get_classification_models_and_data(
    dataset_name: str,
    model_type: str,
    batch_size: int = 512,
    data_path: str = "data",
    seed: int = 0,
):
    project_path = Path(os.path.abspath(""))

    # Convert paths to Path objects consistently
    model_path = (
        project_path
        / "model_checkpoints/best_models"
        / dataset_name
        / model_type
        / f"seed_{seed}"
    )

    # find all the ckpt files in the model_path
    checkpoint_files = list(model_path.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {model_path}")
    model_path = checkpoint_files[0]

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if model_type == "MLP" or model_type == "VQ-VAE_MLP" or model_type == "DVAE_MLP":
        model = MLP.load_from_checkpoint(model_path)
    else:
        raise ValueError(f"Model type {model_type} not supported")

    model.eval()

    # Fix path concatenation
    data_path = Path(data_path)  # Convert to Path object
    val_idx, test_idx = load_val_test_idx(project_path / data_path / dataset_name)

    data_id_path = (
        project_path
        / data_path
        / f"ds_latent_ids_{model_type.split('_')[0]}_{dataset_name}"
    )
    if model_type.startswith("VQ-VAE_"):
        (
            _,
            _,
            _,
            class_train_ds,
            class_val_ds,
            class_test_ds,
        ) = load_all_ds_ids(path=data_id_path)

        data_module = SimpleDataModule(
            train_ds=class_train_ds,
            val_ds=class_val_ds,
            test_ds=class_test_ds,
            batch_size=batch_size,
        )
    else:
        ds, labels, exp_ids = load_raw_data(data_path / dataset_name)

        data_module: WeldingDataModule = WeldingDataModule(
            ds=ds,
            labels=labels,
            exp_ids=exp_ids,
            batch_size=1024,
            shuffle_train=False,
            val_split_idx=val_idx,
            test_split_idx=test_idx,
            n_cycles=1,
            ds_type="classification",
        )
        data_module.setup()
    return model, data_module


def create_new_datasets(data_path: str | Path):
    dataset_names = ["Welding"]
    embedding_model_names = ["VQ-VAE"]
    seq_prediction_tasks = [True, False]
    data_path = Path(data_path)
    for dataset_name in dataset_names:
        for embedding_model_name in embedding_model_names:
            for seq_prediction_task in seq_prediction_tasks:
                vq_vae_path = Path(
                    f"model_checkpoints/best_models/{embedding_model_name}_{dataset_name}.ckpt"
                )
                _ = get_laten_ds(
                    vq_vae_path=vq_vae_path,
                    init_ds=True,
                    n_cycles=1,
                    data_path=data_path,
                    prob_unk_token=0.0,
                    seq_prediction_task=seq_prediction_task,
                )
                print(
                    f"Created new datasets for {dataset_name} with {embedding_model_name} and seq_prediction_task={seq_prediction_task}"
                )


if __name__ == "__main__":
    create_new_datasets("data_docker")
