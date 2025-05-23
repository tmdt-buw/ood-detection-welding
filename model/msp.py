"""
Quelle: https://github.com/hendrycks/error-detection/blob/master/Vision/CIFAR_Detection.py
"""

import logging
import numpy as np
import torch
import mlflow

from ood_score import ood_score_func
from utils_ood import determine_ood_score_threshold
from model.transformer_decoder import MyTransformerDecoder

def msp_ood_detector(
    softmax_prediction: np.ndarray, 
    label_array: np.ndarray, 
    dataset_type: str, 
    calculate_ood_score: bool, 
    threshold: float | None = None,
    use_mlflow: bool = False
) -> float:
    """Detects Out-of-Distribution (OOD) samples using Maximum Softmax
    Probability (MSP).

    This function first identifies whether model predictions are correct or
    incorrect based on the provided labels. It then uses these
    correct/incorrect classifications (treating incorrect predictions as
    pseudo-OOD) to construct a Receiver Operating Characteristic (ROC)
    curve based on the model's prediction confidences (softmax scores).

    An optimal threshold is determined from the ROC curve by maximizing the
    difference between the True Positive Rate (TPR) and False Positive Rate
    (FPR). This threshold is then used to classify samples as In-Distribution
    (ID) or OOD based on their confidence scores. Optionally, it calculates
    and logs an OOD score if requested.

    Args:
        softmax_prediction (np.ndarray): Softmax probability outputs from the
            model for the test dataset. Shape (n_samples, n_classes).
        label_array (np.ndarray): True labels for the test dataset. Shape
            (n_samples,).
        dataset_type (str): Identifier for the dataset being evaluated
            (e.g., "val", "test"). Used for logging purposes.
        calculate_ood_score (bool): If True, calculates and logs the OOD
            score based on the detected OOD samples.
        threshold (float | None, optional): An optional pre-calculated
            threshold. If provided, this threshold is used directly instead
            of calculating a new one from the ROC curve. Defaults to None.

    Returns:
        float: The optimal threshold determined (or used) for OOD detection.
            Samples with confidence below this threshold are classified as OOD.
    """
    if threshold is None:
        pred_classes = np.argmax(softmax_prediction, axis=1)
        optimal_threshold = determine_ood_score_threshold(
            prediction_array=pred_classes,
            label_array=label_array,
            ood_determiner_value=softmax_prediction,
            ood_func="confidence"
        )
    else:
        optimal_threshold = threshold

    logging.info(f"Optimal threshold based on PR curve: {optimal_threshold:.5f}")

    # 3 OOD Detection based on optimal thresold
    confidences = np.max(softmax_prediction, axis=1)
    ood_predictions = confidences < optimal_threshold
    dataset_size = len(confidences)
    logging.info(f"Number of OOD samples in {dataset_type}: {sum(ood_predictions)} [{sum(ood_predictions)}/{dataset_size}] ({sum(ood_predictions) / dataset_size:.2f}%)")
    logging.info(f"Optimal threshold: {optimal_threshold:.5f}")

    if calculate_ood_score:
        pred_classes = np.argmax(softmax_prediction, axis=1)
        quality_predictions = torch.tensor(pred_classes)
        quality_labels = torch.tensor(label_array)

        # Aufteilen in OOD und ID
        predictions_ood = quality_predictions[ood_predictions]
        labels_ood = quality_labels[ood_predictions]

        predictions_id = quality_predictions[~ood_predictions]
        labels_id = quality_labels[~ood_predictions]
        score_f1 = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="f1_score")
        score_acc = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="accuracy")
        logging.info(f"OOD score for {dataset_type}: {score_f1:.2f} (F1), {score_acc:.2f} (Acc)")
        if use_mlflow:
            mlflow.log_metric(f"{dataset_type}/msp/ood_score/f1", score_f1)
            mlflow.log_metric(f"{dataset_type}/msp/ood_score/acc", score_acc)

    return optimal_threshold


def msp(model, data_loader, dataset_type, calculate_ood_score, threshold=None, use_mlflow: bool = False):
    """
    function for preparing MSP based posthoc ood detection by performing predictions with the given model and concatanate numpy arrays for predictions and labels
    Args:
        model: a trained model for quality prediction
        data_loader: dataloader for either the train,val or testset
        dataset_type (string): information about what dataset is used (e.g. "val")
        calculate_ood_score (bool): set True when ood score should be calculated for the given dataset
        threshold (float): optional give a already calculated threshold  

    Returns:
        fload: threshold that was calculated or used
    """
  
    
    logit_list = []
    Y_test = []

    model.eval()
    is_transformer = isinstance(model, MyTransformerDecoder)
    device = model.device
    with torch.no_grad():
        for batch in data_loader:
            if is_transformer:
                input_x, y, _ = batch
            else:
                input_x, y = batch
            input_x = input_x.to(device)
            y = y.to(device)
            logits = model(input_x)
            logit_list.append(logits)
            Y_test.append(y)

    test_logits = torch.cat(logit_list)

    softmax_probs = torch.nn.functional.softmax(test_logits, dim=1)

    threshold = msp_ood_detector(
        softmax_prediction=softmax_probs.cpu().numpy(),
        label_array=torch.cat(Y_test, dim=0).cpu().numpy(),
        dataset_type=dataset_type,
        calculate_ood_score=calculate_ood_score,
        threshold=threshold,
        use_mlflow=use_mlflow
    )
    return threshold

