"""
Applied temporal transformation prediction.

This script trains an autoencoder model to predict temporal transformations on time series data,
which can be used for out-of-distribution (OOD) detection in welding quality monitoring.

Reference:
https://github.com/kaustubhsridhar/time-series-OOD

Example:
    python train_CODiT.py --log saved_models --transformation_list high_pass low_high high_low identity --wl 16
"""
import argparse
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from data_loader.data_module import WeldingDataModule
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch.optim as optim

from utils import load_raw_data
import mlflow
import logging
from tqdm import tqdm

from model.CODiT.models.lenet import Regressor
from model.CODiT.dataset.transformations_batching import GaitBatching
from mlflow_logger import MyMLFlowLogger
from ood_score import ood_score_func

from utils import get_logger, load_val_test_idx
from post_hoc_ood import test_ood_detection
from data_loader.datasets import ClassificationDataset
from torchmetrics import F1Score


def train(
    args: argparse.Namespace,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_dataset: any,
    epoch: int,
) -> None:
    """Train autoencoder to learn predicting temporal transformations on time series data.

    The training process involves:
    1. Constructing a dataset containing original data, transformed data, and transformation IDs
    2. Looping through the dataset (batch-wise), giving the model the original and transformed
       time series as input and predicting which transformation was applied
    3. Calculating loss based on the predicted transformation (logits per class) and
       the actual transformation (ID)

    Args:
        args: Training configuration parameters
        model: Neural network model (regressor from lenet.py)
        criterion: Loss function for calculating transformation and quality prediction losses
        optimizer: Optimization algorithm
        device: Device to run computations on (CPU or GPU)
        train_dataset: Dataset containing training samples
        epoch: Current training epoch
    Returns:
        None
    """

    torch.set_grad_enabled(True)
    model.train()

    dataset = GaitBatching(
        data=train_dataset,
        win_len=args.wl,
        transformation_list=args.transformation_list,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0.0
    correct = 0
    correct_quality = 0

    for orig_data_wins, transformed_data_wins, target_transformations, quality in tqdm(
        dataloader, total=len(dataloader), desc="Training"
    ):
        orig_data_wins = orig_data_wins.to(device)
        transformed_data_wins = transformed_data_wins.to(device)
        target_transformations = target_transformations.to(device)
        quality = quality.to(device)

        optimizer.zero_grad()
        # outputs = model(orig_data_wins, transformed_data_wins) # original forward pass
        outputs, predicted_quality = model.forward_with_quality(
            orig_data_wins, transformed_data_wins
        )  # forward pass with quality

        loss_quality = criterion(predicted_quality.squeeze(), quality)
        loss_transformation = criterion(outputs, target_transformations)
        loss = loss_quality + loss_transformation
        loss.backward()
        optimizer.step()
        # compute loss and acc
        total_loss += loss.item()
        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(target_transformations == pts).item()

        pts_quality = torch.argmax(predicted_quality, dim=1)
        correct_quality += torch.sum(quality == pts_quality).item()

    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / len(dataloader.dataset)
    avg_acc_quality = correct_quality / len(dataloader.dataset)
    if args.use_mlflow:
        mlflow.log_metric("train/loss", avg_loss, step=epoch)
        mlflow.log_metric("train/acc_transformation", avg_acc, step=epoch)
        mlflow.log_metric("train/acc", avg_acc_quality, step=epoch)
    logging.info(
        f"[TRAIN] {epoch}: loss: {avg_loss:.3f}, acc_trans: {avg_acc:.3f}, acc_quality: {avg_acc_quality:.3f}"
    )


def validate(
    args: argparse.Namespace,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    dataset: ClassificationDataset,
    ds_type: Literal["val", "test"],
    epoch: int,
) -> tuple[float, float, float]:
    """Validate model performance during training.

    Args:
        args: Configuration parameters
        model: Neural network model to validate
        criterion: Loss function for calculating transformation and quality prediction losses
        device: Device to run computations on (CPU or GPU)
        dataset: Dataset containing validation or test samples
        ds_type: Specifies if validating on 'val' or 'test' set
        epoch: Current training epoch

    Returns:
        tuple[float, float, float]: Average loss, average quality accuracy, and macro F1 score
    """
    model.eval()
    dataset_loader = GaitBatching(
        data=dataset,
        win_len=args.wl,
        transformation_list=args.transformation_list,
    )
    dataloader = DataLoader(dataset_loader, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    correct = 0
    correct_quality = 0

    num_classes = getattr(dataset, 'num_classes', args.num_classes)

    f1_scorer = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    f1_scorer.reset()

    with torch.no_grad():
        for (
            orig_data_wins,
            transformed_data_wins,
            target_transformations,
            quality,
        ) in dataloader:
            orig_data_wins = orig_data_wins.to(device)
            transformed_data_wins = transformed_data_wins.to(device)
            target_transformations = target_transformations.to(device)
            quality = quality.to(device)

            outputs, predicted_quality = model.forward_with_quality(
                orig_data_wins, transformed_data_wins
            )
            loss_quality = criterion(predicted_quality.squeeze(), quality)
            loss_transformation = criterion(outputs, target_transformations)
            loss = loss_quality + loss_transformation
            total_loss += loss.item()

            pts = torch.argmax(outputs, dim=1)
            correct += torch.sum(target_transformations == pts).item()

            pts_quality = torch.argmax(predicted_quality, dim=1)
            correct_quality += torch.sum(quality == pts_quality).item()

            # Update F1 score
            f1_scorer.update(pts_quality, quality)

    avg_loss = total_loss / len(dataloader)
    acc_transformation = correct / len(dataloader.dataset)
    avg_acc_quality = correct_quality / len(dataloader.dataset)
    f1_score_value = f1_scorer.compute()

    if args.use_mlflow:
        mlflow.log_metric(f"{ds_type}/loss", avg_loss, step=epoch)
        mlflow.log_metric(f"{ds_type}/acc_transformation", acc_transformation, step=epoch)
        mlflow.log_metric(f"{ds_type}/acc", avg_acc_quality, step=epoch)
        # Log F1 score
        mlflow.log_metric(f"{ds_type}/f1_score", f1_score_value.item(), step=epoch)

    logging.info(f"[{ds_type}] Epoch {epoch}: loss: {avg_loss:.3f}, acc: {avg_acc_quality:.3f}, f1_macro: {f1_score_value:.3f}")
    return avg_loss, avg_acc_quality, f1_score_value.item()


def detect_ood(
    args: argparse.Namespace,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    input_dataset: any,
    std_factor: float,
    dataset_type: str,
    calculate_ood_score: bool,
    threshold: float | None = None,
    logger: MyMLFlowLogger | None = None
) -> float:
    """Detect out-of-distribution samples using transformation prediction uncertainty.

    This function is called to either:
    1. Calculate a threshold on validation dataset, or
    2. Apply a pre-calculated threshold to test data to identify OOD samples and calculate OOD scores

    Process:
    1. Get model's predictions for transformations and calculate losses
    2. Calculate average of losses for transformation prediction and construct threshold
       based on standard deviation
    3. Identify samples exceeding the threshold as OOD

    Args:
        args: Configuration parameters
        model: Trained neural network model
        criterion: Loss function that doesn't aggregate the loss of samples for each batch
        device: Device to run computations on (CPU or GPU)
        input_dataset: Dataset to analyze (train, val, or test)
        std_factor: Factor for determining how much over mean+std the transformation
                    loss needs to exceed to be considered OOD
        dataset_type: Dataset type identifier ('train', 'val', or 'test')
        calculate_ood_score: Whether to calculate OOD score metrics
        threshold: Pre-calculated threshold for OOD detection (if None, will be calculated)
        logger: Optional MLflow logger instance for logging metrics
    Returns:
        float: Calculated or applied threshold value
    """
    model.eval()
    dataset = GaitBatching(
        data=input_dataset,
        win_len=args.wl,
        transformation_list=args.transformation_list,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    ood_loss_array = []
    correct = 0
    correct_quality = 0
    labels = []
    predictions = []
    with torch.no_grad():
        for (
            orig_data_wins,
            transformed_data_wins,
            transformation,
            quality,
        ) in dataloader:
            orig_data_wins = orig_data_wins.to(device)
            transformed_data_wins = transformed_data_wins.to(device)
            target_transformations = transformation.clone().detach().to(device)
            # outputs = model(orig_data_wins, transformed_data_wins)
            outputs, predicted_quality = model.forward_with_quality(
                orig_data_wins, transformed_data_wins
            )  # forward pass with quality
            labels.extend(quality)
            predictions.append(predicted_quality)
            loss_transformation = criterion(outputs, target_transformations)
            ood_loss_array.append(loss_transformation)
            pts = torch.argmax(outputs, dim=1)
            correct += torch.sum(target_transformations == pts).item()

            pts_quality = torch.argmax(predicted_quality, dim=1).cpu()
            correct_quality += torch.sum(quality == pts_quality).item()
    summarized = torch.cat(ood_loss_array, dim=0).detach()
    min_loss = 0  # Optional Idea: set to a value over 0 to get a threshold that is only based on the more "Uncertain" data parts, exclude small vacuity samples for mean calculation
    average_loss = torch.mean(summarized[summarized > min_loss]).item()
    std_vacuity = torch.std(summarized).item()
    if threshold is None:  # Calculate threshold, if none is given
        threshold = average_loss + std_factor * std_vacuity
    ood_mask = summarized > threshold
    ood_samples = summarized[ood_mask]
    ood_percentage = len(ood_samples) / len(summarized)
    logging.info(f"OOD Samples in {dataset_type} set: {len(ood_samples)}")
    logging.info(f"OOD percentage in {dataset_type} set: {ood_percentage}")

    if calculate_ood_score:
        predictions = torch.cat(predictions).squeeze()

        pred_classes = torch.argmax(predictions, dim=1)
        labels = torch.tensor(labels).to(device)

        # Split into OOD and ID (in-distribution)
        predictions_ood = pred_classes[ood_mask]
        labels_ood = labels[ood_mask]

        predictions_id = pred_classes[~ood_mask]
        labels_id = labels[~ood_mask]
        acc_score = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="accuracy")
        f1_score = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="f1_score")
        logging.info(f"OOD score for {dataset_type}: {acc_score}")
        if logger:
            logger.log_metric(f"{dataset_type}/CODIT/ood_score/acc", acc_score)
            logger.log_metric(f"{dataset_type}/CODIT/ood_score/f1", f1_score)
            logger.log_metric(f"{dataset_type}/ood_percentage", ood_percentage)

    return threshold

def evaluate_model(
    args: argparse.Namespace,
    best_model_path: Path,
    device: torch.device,
    data_module: WeldingDataModule,
    logger: MyMLFlowLogger | None = None,
    std_factor: float = 1
):
    """Evaluate the trained model for OOD detection performance.

    Loads the best performing model checkpoint and evaluates its OOD
    detection capabilities.
    1. Loads the model specified by `best_model_path`.
    2. Calculates an OOD detection threshold using the validation set based
       on the transformation prediction loss (using `detect_ood`).
    3. Applies the calculated threshold to the training and test sets to
       assess OOD detection performance and log results (using `detect_ood`).

    Args:
        args: Command-line arguments and configuration parameters.
        best_model_path: Path to the saved best model checkpoint.
        device: Device to run computations on (CPU or GPU).
        train_dataset: Dataset containing training samples.
        val_dataset: Dataset containing validation samples.
        test_dataset: Dataset containing test samples.
    """
    # load a trained model first, if not a new one is trained:
    if best_model_path is None:
        raise ValueError("No best model path found")
    else:
        net = Regressor.load_checkpoint(best_model_path, device=device)
        logging.info(f"Loaded model from {best_model_path}")

    criterion_sample_wise = nn.CrossEntropyLoss(reduction="none")
    val_dataset: ClassificationDataset = data_module.val_ds
    test_dataset: ClassificationDataset = data_module.test_ds

    net.eval()
    threshold = detect_ood(
        args=args,
        model=net,
        criterion=criterion_sample_wise,
        device=device,
        input_dataset=val_dataset,
        std_factor=std_factor,
        dataset_type="val",
        calculate_ood_score=False,
        threshold=None,
        logger=logger
    )
    _ = detect_ood(
        args=args,
        model=net,
        criterion=criterion_sample_wise,
        device=device,
        input_dataset=test_dataset,
        std_factor=std_factor,
        dataset_type="test",
        calculate_ood_score=True,
        threshold=threshold,
        logger=logger
    )
    if logger:
        logger.end_run()



def train_model(
    args: argparse.Namespace, 
    data_module: WeldingDataModule, 
    net: nn.Module, 
    mlflow_logger: MyMLFlowLogger | None = None
):
    """
    Train the CODiT model and evaluate its performance.

    This function orchestrates the training process:
    1. Initializes criterion, optimizer, and device.
    2. Iterates through epochs, calling train and validate functions.
    3. Saves the model checkpoint with the best validation loss.
    4. After training, logs the best model artifact if MLflow is used.
    5. Validates the final model on the test set.
    6. Calls `evaluate_model` to perform OOD detection on train, val,
       and test sets using the best model checkpoint.

    Args:
        args: Command-line arguments and configuration parameters.
        data_module: Data module providing datasets.
        net: The neural network model (Regressor).
        mlflow_logger: Optional MLflow logger.
    """
    train_dataset: ClassificationDataset = data_module.train_ds
    val_dataset: ClassificationDataset = data_module.val_ds
    test_dataset: ClassificationDataset = data_module.test_ds

    # cast label to int
    train_dataset.labels = train_dataset.labels.astype(int)
    val_dataset.labels = val_dataset.labels.astype(int)
    test_dataset.labels = test_dataset.labels.astype(int)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        params=net.parameters(), lr=args.lr, weight_decay=args.wgtDecay
    )

    prev_best_val_loss = float("inf")
    save_path = Path(args.save_path) / time.strftime("%Y%m%d_%H%M%S")
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path: Path | None = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Early stopping parameters
    patience = 10
    early_stopping_counter = 0

    # Add placeholder for num_classes if not in args
    if not hasattr(args, 'num_classes'):
        # Infer from dataset or set a default; Example: infer from train_dataset
        # This assumes ClassificationDataset has a way to get num_classes
        args.num_classes = data_module.train_ds.num_classes

    net.to(device) # Ensure model is on the correct device
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        time_start = time.time()
        train(
            args=args,
            model=net,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_dataset=train_dataset,
            epoch=epoch,
        )
        val_loss, _, _ = validate(
            args=args,
            model=net,
            criterion=criterion,
            device=device,
            dataset=val_dataset,
            ds_type="val",
            epoch=epoch
        )

        if val_loss < prev_best_val_loss:
            prev_best_val_loss = val_loss
            best_model_path = save_path / f"epoch_{epoch}.pt"
            net.save_checkpoint(best_model_path)
            logging.info(f"Saved best loss model at epoch {epoch}")
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            # Increment early stopping counter
            early_stopping_counter += 1
            logging.info(f"Early stopping counter: {early_stopping_counter}/{patience}")
            
            # Check if early stopping should be triggered
            if early_stopping_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch} epochs")
                break

    print(f"Best val loss: {prev_best_val_loss}")
    if args.use_mlflow:
        mlflow_logger.log_artifact(best_model_path, "best_model")

    _ = validate(
        args=args,
        model=net,
        criterion=criterion,
        device=device,
        dataset=test_dataset,
        ds_type="test",
        epoch=args.epochs
    )

    if args.compute_ood_score:
        evaluate_model(args, best_model_path, device, data_module, mlflow_logger, std_factor=args.std_factor)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model training and evaluation.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Welding OOD Prediction")
    parser.add_argument("--n-cycles", type=int, default=1, help="number of cycles to consider as input")
    parser.add_argument("--epochs", type=int, default=10, help="number of total epochs to run")
    parser.add_argument("--mode", type=str, default="train", help="train | test")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wgtDecay", default=5e-4, type=float, help="weight decay parameter")
    parser.add_argument("--momentum", type=float, default=9e-1, help="momentum")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path")
    parser.add_argument("--desp", type=str, help="additional description")
    parser.add_argument("--start-epoch", type=int, default=1, help="manual epoch number (useful on restarts)")
    parser.add_argument("--workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--pf", type=int, default=100, help="print frequency every batch")
    parser.add_argument("--seed", type=int, default=100, help="seed for initializing training.")
    parser.add_argument("--transformation_list", nargs="+", default=["dilation", "erosion", "identity"], help="list of transformations to apply to time series data")
    parser.add_argument("--save_path", type=str, default="model_checkpoints/CODiT", help="path to save the model")
    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--use-mlflow", action=argparse.BooleanOptionalAction, default=False, help="Whether to use MLflow")
    parser.add_argument("--compute-ood-score", action=argparse.BooleanOptionalAction, default=True, help="Whether to use OOD detection")
    parser.add_argument("--std-factor", type=float, default=0.25, help="Factor for determining how much over mean+std the transformation loss needs to exceed to be considered OOD")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()
    args.wl = args.n_cycles * 200
    args.transformation_list_len = len(args.transformation_list)
    logging.info(vars(args))

    # Log on MLflfow
    if args.use_mlflow:
        mlflow_logger = MyMLFlowLogger(experiment_name="ood-welding")
        mlflow_logger.start_run()
        mlflow_logger.log_params(vars(args))
    else:
        mlflow_logger = None

    data_path = Path("data/Welding")
    ds, labels, exp_ids = load_raw_data(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed:
        pl.seed_everything(args.seed)

    net = Regressor(
        n_transformation_classes=args.transformation_list_len, 
        seq_len=args.wl, 
        device=device
    )

    val_idx, test_idx = load_val_test_idx(data_path)

    data_module = WeldingDataModule(
        ds,
        labels,
        exp_ids,
        batch_size=args.batch_size,
        val_split_idx=val_idx,
        test_split_idx=test_idx,
        n_cycles=args.n_cycles,
        ds_type="classification"
    )
    
    data_module.setup()
    
    args.num_classes = 2

    if args.mode == "train":
        train_model(args, data_module, net, mlflow_logger)
    elif args.mode == "test":
        evaluate_model(
            args=args,
            best_model_path=Path(args.ckpt),
            device=device,
            data_module=data_module,
            logger=mlflow_logger,
            std_factor=args.std_factor
        )

