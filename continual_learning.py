"""
Module for running continual learning experiments with welding data.

This module implements a continual learning approach where a model is
trained sequentially on different welding experiences.
"""

from pathlib import Path
from datetime import datetime
import logging as log
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Subset, DataLoader
from torch.nn import functional as F
from lightning import seed_everything
from torchmetrics import F1Score

from model.transformer_decoder import MyTransformerDecoder
from data_loader.datasets import (
    MyLatentAutoregressiveDataset,
    MyLatentClassificationDataset,
)
from utils import load_val_test_idx, load_raw_data_experiments, get_laten_ds
from continual_learning.lwf import LearningWithoutForgetting
from continual_learning.replay import (
    ReplayMemory,
    CombinedDataLoader,
    store_samples_in_replay,
)
from vq_vae_transformer_ood import determine_ood_score_threshold, cross_entropy_loss


def load_dataset(
    data_path: Path, n_cycles: int, prob_unk_token: float, seq_prediction_task: bool
) -> tuple[
    MyLatentClassificationDataset,
    MyLatentClassificationDataset,
    MyLatentClassificationDataset,
    MyLatentAutoregressiveDataset,
    MyLatentAutoregressiveDataset,
    np.ndarray,
]:
    """
    Load and prepare datasets for continual learning experiments.

    Args:
        data_path: Path to the data directory
        n_cycles: Number of welding cycles to include in each sample
        prob_unk_token: Probability of replacing tokens with unknown token
        seq_prediction_task: Whether to set up data for sequence prediction

    Returns:
        Tuple containing:
        - Training dataset
        - Validation dataset
        - Test dataset
        - Generation training dataset
        - Generation validation dataset
        - Generation test dataset
        - Array of welding run IDs for test set
    """
    val_idx, test_idx = load_val_test_idx(data_path / "Welding")

    log.info("Loading data")
    ds, labels, exp_ids, welding_run_ids = load_raw_data_experiments(
        data_path / "Welding"
    )

    # filterr welding_run ids
    test_labels = labels[test_idx]
    test_welding_run_ids = welding_run_ids[test_idx]
    test_welding_run_ids = test_welding_run_ids[test_labels != -1]

    vq_vae_path = Path("model_checkpoints/best_models/VQ-VAE/best_vq_vae.ckpt")

    (
        gen_train_ds,
        gen_val_ds,
        gen_test_ds,
        class_train_ds,
        class_val_ds,
        class_test_ds,
    ) = get_laten_ds(
        vq_vae_path=vq_vae_path,
        init_ds=False,
        data_path=data_path,
        n_cycles=n_cycles,
        prob_unk_token=prob_unk_token,
        seq_prediction_task=seq_prediction_task,
    )

    return (
        class_train_ds,
        class_val_ds,
        class_test_ds,
        gen_train_ds,
        gen_val_ds,
        gen_test_ds,
        test_welding_run_ids,
    )


def get_tensor_dataset(ds: MyLatentClassificationDataset) -> TensorDataset:
    """
    Convert a classification dataset to a TensorDataset.

    Args:
        ds: Source classification dataset to convert

    Returns:
        A TensorDataset containing the input features and labels
    """
    one_sample_shape_x = ds[0][0].shape
    one_sample_shape_y = ds[0][1].shape

    x = torch.zeros((len(ds), *one_sample_shape_x), dtype=torch.long)
    y = torch.zeros((len(ds), *one_sample_shape_y), dtype=torch.long)

    is_gen_ds = isinstance(ds, MyLatentAutoregressiveDataset)
    if is_gen_ds:
        one_sample_shape_y_gen = ds[0][2].shape
        y_gen = torch.zeros((len(ds), *one_sample_shape_y_gen), dtype=torch.long)

    for i, data_i in enumerate(ds):
        if is_gen_ds:
            x_i, y_i, y_gen_i = data_i
            y_gen[i] = y_gen_i
        else:
            x_i, y_i = data_i

        x[i] = x_i
        y[i] = y_i

    if is_gen_ds:
        return TensorDataset(x, y, y_gen)
    else:
        return TensorDataset(x, y)


def create_experience_dataloaders(
    ds: TensorDataset, welding_run_ids: np.ndarray, batch_size: int = 256
) -> list[DataLoader]:
    """
    Create separate dataloaders for each welding run experience.

    Args:
        ds: TensorDataset containing all samples
        welding_run_ids: Array of welding run IDs for each sample
        batch_size: Batch size for dataloaders

    Returns:
        List of DataLoaders, one for each unique welding run ID
    """
    experience_datasets = []
    unique_welding_run_ids = np.unique(welding_run_ids)
    for id_welding_run in unique_welding_run_ids:
        subset_ds_idx = np.where(welding_run_ids == id_welding_run)[0]
        subset_dataset = Subset(ds, subset_ds_idx)
        experience_datasets.append(
            DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
        )
    return experience_datasets


def load_model(model_path: Path, device: torch.device | str) -> nn.Module:
    """
    Load a trained model from a checkpoint.

    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model onto (CPU or GPU)

    Returns:
        Loaded PyTorch model
    """
    model: MyTransformerDecoder = MyTransformerDecoder.load_from_checkpoint(checkpoint_path=model_path) # pylint: disable=no-value-for-parameter
    model.to(device)
    return model


def compute_metrics(
    predictions: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """
    Compute evaluation metrics for model predictions.

    Args:
        predictions: Model predictions
        labels: Ground truth labels

    Returns:
        Dictionary containing accuracy and F1 score
    """
    assert (
        predictions.shape == labels.shape
    ), f"predictions.shape: {predictions.shape} != labels.shape: {labels.shape}"
    assert (
        predictions.device == labels.device
    ), f"predictions.device: {predictions.device} != labels.device: {labels.device}"

    acc = (predictions == labels).float().mean()
    f1_scorer = F1Score(task="multiclass", num_classes=2, average="macro").to(
        predictions.device
    )
    f1_scorer.reset()
    f1_scorer.update(predictions, labels)
    f1 = f1_scorer.compute()
    return {"acc": acc.item(), "f1": f1.item()}


def model_forward(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform a forward pass through the model.

    Args:
        model: PyTorch model to use
        batch_x: Input batch of features
        batch_y: Target batch of labels
        device: Device to run the computation on

    Returns:
        Tuple containing loss, predictions, and labels
    """
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    logits = model(batch_x)
    loss = model.loss_cross_entropy(logits, batch_y)
    preds = F.log_softmax(logits, dim=1).argmax(dim=1)
    return loss, preds, batch_y


def evaluate_on_previous_experiences(
    model: torch.nn.Module,
    experience_dataloaders: list[DataLoader],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate model on all previous learning experiences.

    Args:
        model: PyTorch model to evaluate
        experience_dataloaders: List of dataloaders for previous experiences
        device: Device to run the evaluation on

    Returns:
        Tuple of tensors containing all predictions and all labels
    """
    predictions = []
    labels = []
    with torch.no_grad():
        for experience_dataloader in experience_dataloaders:
            for batch_x, batch_y in experience_dataloader:
                _, preds, batch_y = model_forward(model, batch_x, batch_y, device)
                predictions.append(preds)
                labels.append(batch_y)
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    return predictions, labels


def evaluate(
    model: torch.nn.Module,
    continual_learning_dataloaders: list[DataLoader],
    val_dataloader: DataLoader,
    device: torch.device | str,
    result_dict: dict[str, any],
    exp_i: int,
    initial_exp: bool = False,
) -> dict[str, any]:
    """
    Evaluate model on both experiences seen so far and validation data.

    Args:
        model: PyTorch model to evaluate
        continual_learning_dataloaders: List of dataloaders for all experiences
        val_dataloader: Dataloader for validation dataset
        device: Device to run the evaluation on
        result_dict: Dictionary to store evaluation results
        exp_i: Index of the current experience
        initial_exp: Whether the current experience is the initial experience

    Returns:
        Updated result dictionary with evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        exp_predictions, exp_labels = evaluate_on_previous_experiences(
            model, continual_learning_dataloaders[: (exp_i + 1)], device
        )
        exp_metrics = compute_metrics(exp_predictions, exp_labels)
        log.info(
            f"[{exp_i}|{len(continual_learning_dataloaders)}] past experience metrics: {exp_metrics}"
        )
        if initial_exp:
            result_dict["exp_initial_exp_metrics"] = exp_metrics
        else:
            result_dict[f"exp_{exp_i}_exp_metrics"] = exp_metrics

        val_predictions = []
        val_labels = []
        for batch_x, batch_y in val_dataloader:
            _, preds, batch_y = model_forward(model, batch_x, batch_y, device)
            val_predictions.append(preds)
            val_labels.append(batch_y)
        val_predictions = torch.cat(val_predictions)
        val_labels = torch.cat(val_labels)

        val_metrics = compute_metrics(val_predictions, val_labels)
        log.info(
            f"[{exp_i}|{len(continual_learning_dataloaders)}] Validation metrics: {val_metrics}"
        )
        if initial_exp:
            result_dict["exp_initial_exp_val_metrics"] = val_metrics
        else:
            result_dict[f"exp_{exp_i}_val_metrics"] = val_metrics

    return result_dict


def is_ood(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    threshold: float,
) -> bool:
    """
    Determine if a dataset is out-of-distribution (OOD) based on its loss.

    This function evaluates the autoregressive prediction performance of the model
    on the provided dataloader and compares the mean loss against a predetermined
    threshold to identify if the data distribution differs significantly from the
    training distribution.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing data in format (x, _, y_gen)
        device: Device to run the evaluation on
        threshold: Loss threshold above which data is considered OOD

    Returns:
        Boolean indicating whether the data is out-of-distribution
    """
    model.eval()
    loss_items = []
    with torch.no_grad():
        for batch in dataloader:
            x, _, y_gen = batch
            x = x.to(device)
            y_gen = y_gen.to(device)
            # compute ood score
            logits = model(x, generate=True)

            loss_per_item = cross_entropy_loss(
                logits, y_gen, ignore_index=logits.shape[2]
            )
            loss_items.append(loss_per_item.cpu().detach().numpy())
    loss_items = np.concatenate(loss_items, axis=0).mean()
    log.info(f"Loss items: {loss_items} | threshold: {threshold}")
    return loss_items > threshold


def train_autoregressive_head(
    model: MyTransformerDecoder,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device | str,
) -> None:
    """
    Train the autoregressive head of the transformer model.

    This function trains only the autoregressive component of the model,
    which is used for next-token prediction and OOD detection. Training
    this component separately helps maintain accurate OOD detection while
    the model adapts to new classification tasks.

    Args:
        model: The transformer decoder model with an autoregressive head
        optimizer: Optimizer instance to update model parameters
        dataloader: DataLoader containing the training data
        device: Device to run the training on (CPU or GPU)

    Returns:
        None
    """
    model.train()
    for batch in dataloader:
        x, _, y_gen = batch
        x = x.to(device)
        y_gen = y_gen.to(device)

        # compute ood score
        optimizer.zero_grad()
        logits = model(x, generate=True)
        loss = model.loss_cross_entropy(logits, y_gen)

        loss.backward()
        optimizer.step()


def compute_ood_threshold(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device | str,
) -> float:
    """
    Compute a threshold for out-of-distribution detection.

    This function calculates an appropriate threshold for OOD detection
    by analyzing the loss distribution on validation data. It uses the
    determine_ood_score_threshold function to find an optimal cutoff point.

    Args:
        model: PyTorch model to evaluate
        val_loader: DataLoader containing validation data
        device: Device to run the computation on

    Returns:
        Float value representing the OOD detection threshold
    """
    model.eval()
    val_loss = []
    val_preds = []
    val_labels = []
    # build ood threshold on validation dataset
    with torch.no_grad():
        for batch in val_loader:
            x, y, y_gen = batch
            x = x.to(device)
            y_gen = y_gen.to(device)
            # compute ood score
            logits = model(x, generate=True)
            # compute loss
            loss_per_item = cross_entropy_loss(
                logits, y_gen, ignore_index=logits.shape[2]
            )
            val_loss.append(loss_per_item.cpu().detach().numpy())

            logits = model(x, generate=False)
            preds = F.log_softmax(logits, dim=1).argmax(dim=1)

            val_preds.append(preds)
            val_labels.append(y)

    val_loss = np.concatenate(val_loss, axis=0)
    val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
    val_labels = torch.cat(val_labels, dim=0).cpu().numpy()

    val_loss_filter = val_loss < np.mean(val_loss) + 3 * np.std(val_loss)
    val_preds_filtered = val_preds[val_loss_filter]
    val_labels_filtered = val_labels[val_loss_filter]
    val_loss_filtered = val_loss[val_loss_filter]

    threshold = determine_ood_score_threshold(
        prediction_array=val_preds_filtered,
        label_array=val_labels_filtered,
        ood_determiner_value=val_loss_filtered,
        ood_func="error",
    )
    log.info(f"OOD threshold: {threshold:.4f}")
    return threshold


def continual_learning_loop(
    model: torch.nn.Module,
    continual_learning_dataloaders: list[DataLoader],
    val_dataloader: DataLoader,
    gen_train_dataloader: DataLoader,
    gen_val_dataloader: DataLoader,
    gen_test_dataloaders: list[DataLoader],
    old_train_dataloader: DataLoader,
    device: torch.device | str,
    conf: dict[str, any],
    use_continual_learning: bool,
    use_ood_detection: bool,
    use_replay: bool,
) -> dict[str, any]:
    """
    Run the continual learning loop on a sequence of experiences.

    The model is trained sequentially on different experiences (welding runs)
    and evaluated after each experience on both previous experiences and
    validation data. This implementation supports multiple continual learning
    techniques including Learning without Forgetting (LwF), Experience Replay,
    and OOD detection.

    Args:
        model: PyTorch model to train
        continual_learning_dataloaders: List of dataloaders for experiences
        val_dataloader: Dataloader for validation dataset
        gen_train_dataloader: Dataloader for generation training dataset
        gen_val_dataloader: Dataloader for generation validation dataset
        gen_test_dataloaders: List of dataloaders for test generation
        old_train_dataloader: Dataloader for training dataset
        device: Device to run training on
        conf: Configuration dictionary with hyperparameters
        use_continual_learning: Whether to use Learning without Forgetting
        use_ood_detection: Whether to use OOD detection
        use_replay: Whether to use experience replay

    Returns:
        Dictionary containing all evaluation results
    """
    optimizer = model.configure_optimizers()

    epochs_continual_learning = conf["epochs_continual_learning"]
    result_dict = {}
    log.info("Starting continual learning loop")

    # Initialize LwF
    lwf = None
    if use_continual_learning:
        lwf = LearningWithoutForgetting(
            model=model,
            temperature=conf.get("lwf_temperature", 2.0),
            lambda_old=conf.get("lwf_lambda_old", 1.0),
            lambda_new=conf.get("lwf_lambda_new", 1.0),
        )

    # Initialize Replay Memory
    replay_memory = None
    if use_replay:
        replay_memory = ReplayMemory(
            capacity=conf.get("replay_memory_capacity", 1000),
            sample_selection=conf.get("replay_sample_selection", "random"),
            task_balanced=conf.get("replay_task_balanced", True),
            device=device,
            is_gen_dataset=False,
        )

        replay_gen_memory = ReplayMemory(
            capacity=conf.get("replay_memory_capacity", 1000),
            sample_selection=conf.get("replay_sample_selection", "random"),
            task_balanced=conf.get("replay_task_balanced", True),
            device=device,
            is_gen_dataset=True,
        )

    # Initial validation
    result_dict_exp = evaluate(
        model,
        continual_learning_dataloaders,
        val_dataloader,
        device,
        result_dict,
        len(continual_learning_dataloaders),
        initial_exp=True,
    )
    result_dict.update(result_dict_exp)

    if use_replay:
        store_samples_in_replay(
            replay_memory, old_train_dataloader, "experience_initial"
        )
        store_samples_in_replay(
            replay_gen_memory, gen_train_dataloader, "experience_initial"
        )

    for exp_i, experience_dataloader in enumerate(continual_learning_dataloaders):
        model.train()

        if use_ood_detection:
            ood_threshold = compute_ood_threshold(model, gen_val_dataloader, device)
            is_ood_experience = is_ood(
                model, gen_test_dataloaders[exp_i], device, ood_threshold
            )
            result_dict[f"exp_{exp_i}_ood_experience"] = 1 if is_ood_experience else 0
            if not is_ood_experience:
                log.info(f"--> Experience {exp_i} is not OOD, skipping")
                result_dict_exp = evaluate(
                    model,
                    continual_learning_dataloaders,
                    val_dataloader,
                    device,
                    result_dict,
                    exp_i,
                )
                result_dict.update(result_dict_exp)

                # Store samples in replay memory even if we're skipping training
                if use_replay:
                    store_samples_in_replay(
                        replay_memory, experience_dataloader, task_id
                    )

                continue
            else:
                log.info(f"--> Experience {exp_i} is OOD, training on it")

        # Get replay dataloader if using replay
        replay_dataloader = None
        replay_gen_dataloader = None
        if use_replay and exp_i > 0:
            replay_dataloader = replay_memory.get_replay_dataloader(
                batch_size=conf["batch_size"], exclude_task_id=task_id
            )
            replay_gen_dataloader = replay_gen_memory.get_replay_dataloader(
                batch_size=conf["batch_size"], exclude_task_id=task_id
            )
            if replay_dataloader is not None:
                log.info(f"Using replay with {len(replay_dataloader.dataset)} samples")

        # Create combined dataloader that interleaves current experience and replay samples
        combined_dataloader = CombinedDataLoader(
            current_dataloader=experience_dataloader,
            replay_dataloader=replay_dataloader,
            batch_size=conf["batch_size"],
        )
        log.info(f"Created combined dataloader with {len(combined_dataloader)} batches")

        if use_ood_detection:
            combined_gen_dataloader = CombinedDataLoader(
                current_dataloader=gen_test_dataloaders[exp_i],
                replay_dataloader=replay_gen_dataloader,
                batch_size=conf["batch_size"],
            )
            log.info(
                f"Created combined gen dataloader with {len(combined_gen_dataloader)} batches"
            )
            train_autoregressive_head(model, optimizer, combined_gen_dataloader, device)

        task_id = f"experience_{exp_i}"
        if use_continual_learning:
            lwf.record_old_task_outputs(task_id, combined_dataloader, device)

        for _ in range(epochs_continual_learning):
            # Train on interleaved batches from current experience and replay memory
            for b_i, batch in enumerate(combined_dataloader):
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)
                new_loss = model.loss_cross_entropy(logits, batch_y)

                # Apply LwF if enabled
                if use_continual_learning:
                    loss = lwf.compute_combined_loss(new_loss, task_id, batch_x, b_i)
                else:
                    loss = new_loss

                loss.backward()
                optimizer.step()

        # Store samples from current experience in replay memory
        if use_replay:
            store_samples_in_replay(replay_memory, experience_dataloader, task_id)

        result_dict_exp = evaluate(
            model,
            continual_learning_dataloaders,
            val_dataloader,
            device,
            result_dict,
            exp_i,
        )
        result_dict.update(result_dict_exp)

    return result_dict


def main():
    """
    Main function to run the continual learning experiment.

    Sets up the experiment configuration, loads data and model, and runs
    the continual learning loop. Results are saved to a JSON file.
    """
    prob_unk_token = 0.0
    seq_prediction_task = False
    seed = 42
    batch_size = 512
    learning_rate = 0.001
    epochs_continual_learning = 4
    use_continual_learning = True
    use_ood_detection = True
    use_replay = True
    model_path = Path(
        "model_checkpoints/best_models/VQ-VAE_Transformer/best_transformer.ckpt"
    )

    # LwF hyperparameters
    lwf_temperature = 2.0
    lwf_lambda_old = 1.0
    lwf_lambda_new = 1.0

    # Replay hyperparameters
    replay_memory_capacity = 5_000
    replay_sample_selection = "random"  # options: "random", "loss", "entropy"
    replay_task_balanced = True

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    n_cycles = model.n_cycles
    conf = {
        "epochs_continual_learning": epochs_continual_learning,
        "learning_rate": learning_rate,
        "seed": seed,
        "batch_size": batch_size,
        "n_cycles": n_cycles,
        "prob_unk_token": prob_unk_token,
        "seq_prediction_task": seq_prediction_task,
        "lwf_temperature": lwf_temperature,
        "lwf_lambda_old": lwf_lambda_old,
        "lwf_lambda_new": lwf_lambda_new,
        "replay_memory_capacity": replay_memory_capacity,
        "replay_sample_selection": replay_sample_selection,
        "replay_task_balanced": replay_task_balanced,
    }

    data_path = Path("data")

    log.info("Loading dataset")
    (
        class_train_ds,
        class_val_ds,
        class_test_ds,
        gen_train_ds,
        gen_val_ds,
        gen_test_ds,
        test_welding_run_ids,
    ) = load_dataset(data_path, n_cycles, prob_unk_token, seq_prediction_task)

    # Create a dataloader for training data (used by LwF)
    train_dataloader = DataLoader(
        get_tensor_dataset(class_train_ds), batch_size=batch_size, shuffle=True
    )

    gen_train_dataloader = DataLoader(
        get_tensor_dataset(gen_train_ds), batch_size=batch_size, shuffle=True
    )

    val_dataloader = DataLoader(
        get_tensor_dataset(class_val_ds), batch_size=batch_size, shuffle=False
    )

    gen_val_dataloader = DataLoader(gen_val_ds, batch_size=batch_size, shuffle=False)

    gen_test_dataset = get_tensor_dataset(gen_test_ds)

    gen_test_dataloaders = create_experience_dataloaders(
        gen_test_dataset, test_welding_run_ids, batch_size
    )

    test_tensor_dataset = get_tensor_dataset(class_test_ds)

    continual_learning_dataloaders = create_experience_dataloaders(
        test_tensor_dataset, test_welding_run_ids, batch_size
    )

    log.info("Done loading dataset")
    log.info(
        f"continual_learning_dataloaders: {len(continual_learning_dataloaders)} type: {type(continual_learning_dataloaders[0])}"
    )

    result_dict = continual_learning_loop(
        model=model,
        continual_learning_dataloaders=continual_learning_dataloaders,
        val_dataloader=val_dataloader,
        gen_train_dataloader=gen_train_dataloader,
        gen_val_dataloader=gen_val_dataloader,
        gen_test_dataloaders=gen_test_dataloaders,
        old_train_dataloader=train_dataloader,
        device=device,
        conf=conf,
        use_continual_learning=use_continual_learning,
        use_ood_detection=use_ood_detection,
        use_replay=use_replay,
    )

    log.info(result_dict)

    str_use_continual_learning = "CL" if use_continual_learning else "noCL"
    str_use_ood_detection = "OOD" if use_ood_detection else "noOOD"
    str_use_replay = "Replay" if use_replay else "noReplay"
    file_name = f"results_{str_use_continual_learning}_{str_use_ood_detection}_{str_use_replay}_{str_use_replay}_epochs{epochs_continual_learning}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    with open(f"results/{file_name}", "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)

    log.info("Done")


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
