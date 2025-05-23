import logging
from typing import Literal

import numpy as np
from sklearn.metrics import roc_curve
import torch
from torch import Tensor

def confidence_fn(ood_determiner_value: np.ndarray, correct_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates confidence scores and correctness labels for ROC analysis.

    This function uses the maximum value of the softmax prediction as the
    confidence score. It separates the confidence scores based on whether
    the corresponding prediction was correct and returns arrays suitable
    for ROC curve calculation.

    Args:
        ood_determiner_value: A 2D array where each row is the softmax
            output for a prediction.
        correct_predictions: A boolean array where True indicates a correct
            prediction and False indicates an incorrect prediction.

    Returns:
        A tuple containing:
            - correctness_labels (np.ndarray): An array of labels
              (1 for correct, 0 for incorrect).
            - confidences (np.ndarray): The corresponding confidence scores,
              ordered according to the correctness_labels.
    """
    confidence_value = np.max(ood_determiner_value, axis=1)
   
    confidence_right = confidence_value[correct_predictions]  # confidences for only correct predcitions
    confidence_wrong = confidence_value[~correct_predictions] 

    correctness_labels = np.concatenate([np.ones(len(confidence_right)), np.zeros(len(confidence_wrong))])
    confidences = np.concatenate([confidence_right, confidence_wrong])

    return correctness_labels, confidences



def error_fn(ood_determiner_value: np.ndarray, correct_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares OOD determiner values and correctness labels for ROC analysis.

    This function takes an array of pre-computed OOD determiner values
    (e.g., reconstruction error) and a boolean array indicating whether
    each corresponding prediction was correct. It separates the OOD values
    based on correctness and returns arrays suitable for ROC curve
    calculation.

    Args:
        ood_determiner_value: An array of values from an OOD detection
            method (e.g., reconstruction error).
        correct_predictions: A boolean array where True indicates a correct
            prediction and False indicates an incorrect prediction.

    Returns:
        A tuple containing:
            - error_labels (np.ndarray): An array of labels (1 for correct,
              0 for incorrect).
            - ood_values (np.ndarray): The corresponding OOD determiner
              values, ordered according to the error_labels.
    """
    ood_values_right = ood_determiner_value[correct_predictions]
    ood_values_wrong = ood_determiner_value[~correct_predictions]

    error_labels = np.concatenate([np.ones(len(ood_values_right)), np.zeros(len(ood_values_wrong))])
    ood_values = np.concatenate([ood_values_right, ood_values_wrong])

    return error_labels, ood_values


ood_func_dict = {
    "confidence": confidence_fn,
    "error": error_fn,
}

def determine_ood_score_threshold(
    prediction_array: np.ndarray,
    label_array: np.ndarray,
    ood_determiner_value: np.ndarray,
    ood_func: Literal["confidence", "error"]
) -> float:
    """
    Determines the optimal OOD score threshold using ROC analysis.

    Calculates the optimal threshold for distinguishing between correct and
    incorrect predictions based on a given OOD score (either prediction
    confidence or an external value like reconstruction error). The optimal
    threshold is found by maximizing the Youden's J statistic (TPR - FPR)
    on the ROC curve.

    Args:
        prediction_array: An array of model predictions.
        label_array: An array of true labels corresponding to the predictions.
        ood_determiner_value: An array of values to be used for OOD
            detection. This could be softmax confidences (if ood_func is
            'confidence') or other values like reconstruction errors (if
            ood_func is 'error').
        ood_func: A literal string specifying which function to use for
            calculating OOD scores and labels ('confidence' or 'error').

    Returns:
        The optimal threshold value determined from the ROC curve.
    """
    correct_predictions = prediction_array == label_array

    ood_correctness_labels, ood_values = ood_func_dict[ood_func](
        ood_determiner_value=ood_determiner_value,
        correct_predictions=correct_predictions
    )

    fpr, tpr, thresholds_roc = roc_curve(
        ood_correctness_labels, ood_values
    )  

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds_roc[max(1, optimal_idx)]
    logging.info(f"Optimal threshold based on PR curve: {optimal_threshold:.5f}")

    return optimal_threshold

def cross_entropy_loss(
    logits: Tensor, labels: Tensor, ignore_index: int = -100
) -> Tensor:
    """
    Calculates the cross-entropy loss given logits and labels, with an option to ignore certain indices.

    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, seq_len, num_classes)
        labels (torch.Tensor): Tensor of shape (batch_size, seq_len), containing class indices
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.

    Returns:
        torch.Tensor: The mean cross-entropy loss for the batch, ignoring specified indices
    """
    # Flatten logits and labels to shape (-1, num_classes) and (-1) respectively
    batch_size, seq_len, num_classes = logits.shape
    logits = logits.view(batch_size * seq_len, num_classes)
    labels = labels.view(batch_size * seq_len)  # -1

    # Mask to identify non-ignored indices
    valid_mask = labels != ignore_index

    # Apply log-softmax to the logits
    log_probs = torch.log_softmax(logits, dim=-1)

    # Select the log-probabilities corresponding to the correct labels, only for valid indices
    loss = -log_probs[torch.arange(log_probs.size(0)), labels]

    # Mask the loss for ignored indices and compute the mean of non-ignored losses
    loss = loss * valid_mask

    # mean_loss = loss.sum() / valid_mask.sum()

    reshaped_loss = loss.view(batch_size, seq_len)
    reshaped_loss = reshaped_loss.sum(dim=1)
    reshaped_loss = reshaped_loss / valid_mask.sum()

    return reshaped_loss