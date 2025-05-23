import logging as log
from typing import Literal

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import mlflow

from model.mlp import MLP
from model.msp import msp
from model.ODIN import ODIN
from model.mahalanobis import mahalanobis_detector
from model.transformer_decoder import MyTransformerDecoder
from ood_score import ood_score_func
from data_loader.data_module import WeldingDataModule



def vacuity_mean_std_based_ood_detection(
    model: MLP,
    dataloader: DataLoader,
    dataset_type: Literal["train", "val", "test"],
    std_factor: float,
    calculate_ood_score: bool,
    threshold: float | None = None,
    use_mlflow: bool = False
) -> float:
    """
    ood detection based on the vacuity component, cacluclating mean vacuity, and detects ood if vacuity exceeds mean+std_factor*std
    and writes results to a csv file (results/vacuity_based_detection.csv)
    Args:
        model: trained quality prediction model
        dataloader: loader for quality data
        dataset_type (string): train, val or test
        std_factor (float): factor determining to what extend vacutiy values need to be over the average to be detected as ood
        calculate_ood_score (bool): determines if the ood score is calculated for the given dataset
        threshold (float): optional give threshold if already calculated
        use_mlflow (bool): whether to use mlflow for logging
    returns:
        float: threshold for ood detection
    """
    model.train()
    vacuity = []
    predictions = []
    labels = []
    is_transformer = isinstance(model, MyTransformerDecoder)
    device = model.device
    for batch in dataloader:
        if is_transformer:
            input_x, y, _ = batch
            input_x = input_x.to(device)
            y = y.to(device)
            logits = model(input_x, generate=False)
        else:
            input_x, y = batch
            input_x = input_x.to(device)
            y = y.to(device)
            logits = model(input_x)
        preds = F.log_softmax(logits, dim=1).argmax(dim=1)
        labels.extend(y.detach().cpu())
        predictions.extend(preds)
        # Calculate and append vacuity for the prediction
        alpha = torch.exp(logits) + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        S_0_removed = S[S != 0]
        vacuity_tmp = 2 / S_0_removed  # 2 --> because binary classsification
        vacuity_tmp = vacuity_tmp[~torch.isnan(vacuity_tmp)]
        vacuity_tmp = vacuity_tmp.detach()
        vacuity.append(vacuity_tmp)

    summarized = torch.cat(vacuity, dim=0).detach()
    min_vacuity = 0  # Optional Idea: set to a value over 0 to get a threshold that is only based on the more "Uncertain" data parts, exclude small vacuity samples for mean calculation
    average_vacuity = torch.mean(summarized[summarized > min_vacuity]).item()
    std_vacuity = torch.std(summarized).item()
    if threshold is None:  # Calcluate threshold, if none is given
        threshold = average_vacuity + std_factor * std_vacuity
    ood_mask = summarized > threshold
    ood_samples = summarized[ood_mask]
    ood_percentage = len(ood_samples) / len(summarized)
    log.info(f"OOD Samples in {dataset_type} set: [{len(ood_samples)} | {len(summarized)}] ({ood_percentage:.2f}%)")

    if calculate_ood_score:
        predictions = torch.tensor(predictions, device=device)
        labels = torch.tensor(labels, device=device)

        # Aufteilen in OOD und ID
        predictions_ood = predictions[ood_mask]
        labels_ood = labels[ood_mask]

        predictions_id = predictions[~ood_mask]
        labels_id = labels[~ood_mask]
        score_f1 = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="f1_score")
        score_acc = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="accuracy")
        log.info(f"OOD score for {dataset_type}: {score_f1:.2f} (F1), {score_acc:.2f} (Acc)")
        if use_mlflow:
            mlflow.log_metric(f"{dataset_type}/vacuity_based/ood_score/f1", score_f1)
            mlflow.log_metric(f"{dataset_type}/vacuity_based/ood_score/acc", score_acc)


    return threshold



def vacuity_based_detection(model: MLP, val_loader: DataLoader, test_loader: DataLoader, std_factor: float = 1, use_mlflow: bool = False):
    """
    Performs out-of-distribution detection using vacuity-based approach across validation and test datasets.

    Args:
        model (MLP): The trained neural network model
        val_loader: DataLoader for the validation dataset
        test_loader: DataLoader for the test dataset
        std_factor (float, optional): Factor determining the threshold as mean + std_factor * std. Defaults to 1.
        use_mlflow (bool, optional): Whether to use MLFlow for logging. Defaults to False.
    Returns:
        None: Results are saved internally
    """
    model.eval()
    threshold = vacuity_mean_std_based_ood_detection(
        model=model,
        dataloader=val_loader,
        dataset_type="val",
        std_factor=std_factor,
        calculate_ood_score=False,
    )
    _ = vacuity_mean_std_based_ood_detection(
        model=model,
        dataloader=test_loader,
        dataset_type="test",
        std_factor=std_factor,
        calculate_ood_score=True,
        threshold=threshold,
        use_mlflow=use_mlflow
    )

def msp_ood_detection(model: MLP, val_loader: DataLoader, test_loader: DataLoader, use_mlflow: bool = False):
    """
    Performs out-of-distribution detection using Maximum Softmax Probability (MSP) approach.

    Args:
        model (MLP): The trained neural network model
        val_loader: DataLoader for the validation dataset
        test_loader: DataLoader for the test dataset
        use_mlflow (bool, optional): Whether to use MLFlow for logging. Defaults to False.
    Returns:
        None: Results are saved internally
    """
    model.eval()
    threshold = msp(model, val_loader, "val", calculate_ood_score=False)

    _ = msp(model, test_loader, "test", calculate_ood_score=True, threshold=threshold, use_mlflow=use_mlflow)

def odin_ood_detection(
    model: MLP,
    val_loader,
    test_loader,
    temperature=1000,
    magnitude=0.0014,
    use_mlflow: bool = False
):
    """
    Performs out-of-distribution detection using ODIN (Out-of-DIstribution detector for Neural networks) approach.

    Args:
        model (MLP): The trained neural network model
        val_loader: DataLoader for the validation dataset
        test_loader: DataLoader for the test dataset
        temperature (float, optional): Temperature scaling parameter. Defaults to 1000.
        magnitude (float, optional): Perturbation magnitude parameter. Defaults to 0.0014.

    Returns:
        None: Results are saved internally
    """
    threshold = ODIN(
        model,
        val_loader,
        "val",
        temperature,
        magnitude,
        calculate_ood_score=False,
    )

    _ = ODIN(
        model,
        test_loader,
        "test",
        temperature,
        magnitude,
        calculate_ood_score=True,
        threshold=threshold,
        use_mlflow=use_mlflow
    )


def mahalanobis_ood_detection(
    model, data_module, quality_based=False, magnitude=0.0014, temperature=1000, use_mlflow: bool = False
):
    """
    Performs out-of-distribution detection using Mahalanobis distance-based approach.

    This method calculates class-conditional Gaussian distributions and uses the Mahalanobis
    distance to identify samples that differ significantly from the training distribution.

    Args:
        model: The trained neural network model
        data_module: Module containing the data loaders and dataset information
        quality_based (bool, optional): If True, detection is based on quality assumption.
                                       If False, detection is based on false detection anomalies.
                                       Defaults to False.
        magnitude (float, optional): Perturbation magnitude parameter. Defaults to 0.0014.
        temperature (float, optional): Temperature scaling parameter. Defaults to 1000.
        use_mlflow (bool, optional): Whether to use MLFlow for logging. Defaults to False.
    Returns:
        None: Results are processed internally by the mahalanobis_detector function
    """
    mahalanobis_detector(
        model,
        trainloader=data_module.train_dataloader(),
        valloader=data_module.val_dataloader(),
        num_classes=2,
        batch_size=data_module.batch_size,
        testloader=data_module.test_dataloader(),
        train_size=len(data_module.train_ds),
        val_size=len(data_module.val_ds),
        quality_based=quality_based,
        magnitude=magnitude,
        temperature=temperature,
        use_mlflow=use_mlflow
    )


def test_ood_detection(model, data_module: WeldingDataModule, use_mlflow: bool):
    """Runs posthoc OOD detection methods on the provided model and data. The following methods are tested:
    - Vacuity Mean Std
    - MSP
    - Mahalanobis
    - ODIN

    Args:
        model: The trained model to evaluate.
        data_module: The data module containing validation and test loaders.
        use_mlflow (bool): Whether to log results using MLflow.
    """
    test_loader = data_module.test_dataloader()
    val_loader = data_module.val_dataloader()

    model.eval()

    log.info("--------------------------------")
    log.info("OOD Detection using Vacuity Mean Std")
    vacuity_based_detection(
        model=model, 
        val_loader=val_loader, 
        test_loader=test_loader,
        use_mlflow=use_mlflow
    )

    log.info("--------------------------------")
    log.info("OOD Detection using MSP")
    msp_ood_detection(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        use_mlflow=use_mlflow
    )
    log.info("--------------------------------")

    log.info("OOD Detection using ODIN")
    odin_ood_detection(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        use_mlflow=use_mlflow
    )
    log.info("--------------------------------")

    log.info("OOD Detection using Mahalanobis")
    mahalanobis_ood_detection(
        model=model, 
        data_module=data_module, 
        use_mlflow=use_mlflow
    )
    log.info("--------------------------------")
