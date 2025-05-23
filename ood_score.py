import torch
from torch import Tensor
import numpy as np
from typing import Literal
import logging as log
from sklearn.metrics import accuracy_score, f1_score as f1


def ood_score_func(
    in_dist_pred: Tensor,
    out_dist_pred: Tensor,
    in_dist_labels: Tensor,
    out_dist_labels: Tensor,
    metric: Literal["f1_score", "accuracy"] = "accuracy",
    beta: float = 0.5,
) -> float:
    """
    Score for OOD detection. The higher the score, the better the model is at detecting OOD samples.

    OOD score = (in_dist_score - out_dist_score)
    The score ranges from [-1, 1]

    Perfect In-Distribution, Worst Out-of-Distribution -> 1
    Perfect In-Distribution, Perfect Out-of-Distribution -> 0
    Worst In-Distribution, Perfect Out-of-Distribution -> -1


    Args:
        in_dist_pred (Tensor): Predictions for in-distribution samples
        out_dist_pred (Tensor): Predictions for out-of-distribution
        in_dist_labels (Tensor): Labels for in-distribution samples
        out_dist_labels (Tensor): Labels for out-of-distribution samples
        metric (Literal["f1_score", "accuracy"], optional): Metric to use for computing the score. Defaults to "accuracy".
        beta (float, optional): Beta parameter for the OOD score. Defaults to 0.75.
    Returns:
        float: OOD score
    """

    # to cpu & numpy
    in_dist_pred = in_dist_pred.cpu().numpy()
    out_dist_pred = out_dist_pred.cpu().numpy()
    in_dist_labels = in_dist_labels.cpu().numpy()
    out_dist_labels = out_dist_labels.cpu().numpy()

    # print(f"{in_dist_pred.shape=}, {out_dist_pred.shape=}, {in_dist_labels.shape=}, {out_dist_labels.shape=}")

    if metric == "f1_score":
        score_in_dist = f1(y_true=in_dist_labels, y_pred=in_dist_pred, average="macro")
        score_out_dist = f1(
            y_true=out_dist_labels, y_pred=out_dist_pred, average="macro"
        )
    elif metric == "accuracy":
        score_in_dist = accuracy_score(y_true=in_dist_labels, y_pred=in_dist_pred)
        score_out_dist = accuracy_score(y_true=out_dist_labels, y_pred=out_dist_pred)
    else:
        raise ValueError(f"Metric {metric} not supported")
    ood_score = (beta * score_in_dist + score_in_dist - score_out_dist) / (1 + beta)

    log.info(
        f"score_in_dist: {score_in_dist:.3f}, score_out_dist: {score_out_dist:.3f} | score diff: {(score_in_dist - score_out_dist):.3f} | beta: {beta:.3f} | ood_score: {ood_score:.2f}"
    )
    return ood_score


def ood_score_func_np(
    in_dist_pred: np.ndarray,
    out_dist_pred: np.ndarray,
    in_dist_labels: np.ndarray,
    out_dist_labels: np.ndarray,
    metric: Literal["f1_score", "accuracy"] = "accuracy",
    beta: float = 0.75,
) -> float:
    """
    Score for OOD detection. The higher the score, the better the model is at detecting OOD samples.

    OOD score = (in_dist_score - out_dist_score)
    The score ranges from [-1, 1]

    Perfect In-Distribution, Worst Out-of-Distribution -> 1
    Perfect In-Distribution, Perfect Out-of-Distribution -> 0
    Worst In-Distribution, Perfect Out-of-Distribution -> -1

    Args:
        in_dist_pred (np.ndarray): Predictions for in-distribution samples
        out_dist_pred (np.ndarray): Predictions for out-of-distribution
        in_dist_labels (np.ndarray): Labels for in-distribution samples
        out_dist_labels (np.ndarray): Labels for out-of-distribution samples
        metric (Literal["f1_score", "accuracy"], optional): Metric to use for computing the score. Defaults to "accuracy".
        beta (float, optional): Beta parameter for the OOD score. Defaults to 0.75.
    Returns:
        float: OOD score
    """

    # print(f"{in_dist_pred.shape=}, {out_dist_pred.shape=}, {in_dist_labels.shape=}, {out_dist_labels.shape=}")

    if metric == "f1_score":
        score_in_dist = f1(y_true=in_dist_labels, y_pred=in_dist_pred, average="macro")
        score_out_dist = f1(
            y_true=out_dist_labels, y_pred=out_dist_pred, average="macro"
        )
    elif metric == "accuracy":
        score_in_dist = accuracy_score(y_true=in_dist_labels, y_pred=in_dist_pred)
        score_out_dist = accuracy_score(y_true=out_dist_labels, y_pred=out_dist_pred)
    ood_score = (beta * score_in_dist + score_in_dist - score_out_dist) / (1 + beta)

    log.info(
        f"score_in_dist: {score_in_dist:.5f}, score_out_dist: {score_out_dist:.5f} | score_difference: {ood_score:.5f}"
    )
    return ood_score


def test_ood_score():
    # Case 1: Perfect classification on both in-distribution and out-of-distribution (expect score of 0)
    in_dist_labels = torch.tensor([0, 1, 0, 1])
    in_dist_pred = torch.tensor([0, 1, 0, 1])  # Perfect predictions
    out_dist_labels = torch.tensor([0, 1, 0, 1])
    out_dist_pred = torch.tensor([0, 1, 0, 1])  # Also perfect predictions
    print("Case 1 - Perfect Classification on Both:")
    print(
        f"OOD Score (Accuracy): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='accuracy')}"
    )
    print(
        f"OOD Score (F1 Score): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='f1_score')}\n"
    )

    # Case 2: Perfect in-distribution, worst out-of-distribution
    in_dist_labels = torch.tensor([0, 1, 0, 1])
    in_dist_pred = torch.tensor([0, 1, 0, 1])  # Perfect in-distribution predictions
    out_dist_labels = torch.tensor([0, 1, 0, 1])
    out_dist_pred = torch.tensor([1, 0, 1, 0])  # Opposite predictions
    print("Case 2 - Perfect In-Distribution, Worst Out-of-Distribution:")
    print(
        f"OOD Score (Accuracy): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='accuracy')}"
    )
    print(
        f"OOD Score (F1 Score): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='f1_score')}\n"
    )

    # Case 3: Worst in-distribution, perfect out-of-distribution
    in_dist_labels = torch.tensor([0, 1, 0, 1])
    in_dist_pred = torch.tensor([1, 0, 1, 0])  # Opposite in-distribution predictions
    out_dist_labels = torch.tensor([0, 1, 0, 1])
    out_dist_pred = torch.tensor([0, 1, 0, 1])  # Perfect OOD predictions
    print("Case 3 - Worst In-Distribution, Perfect Out-of-Distribution:")
    print(
        f"OOD Score (Accuracy): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='accuracy')}"
    )
    print(
        f"OOD Score (F1 Score): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='f1_score')}\n"
    )

    # Case 4: Zero Division (mean_score is zero)
    in_dist_labels = torch.tensor([0, 0])
    in_dist_pred = torch.tensor([1, 1])  # Incorrect predictions
    out_dist_labels = torch.tensor([1, 1])
    out_dist_pred = torch.tensor([0, 0])  # Incorrect predictions
    print("Case 4 - Zero Division (mean_score is zero):")
    print(
        f"OOD Score (Accuracy): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='accuracy')}"
    )
    print(
        f"OOD Score (F1 Score): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='f1_score')}\n"
    )

    # Case 5: Small and large tensor sizes
    # Minimal size (single element)
    in_dist_labels = torch.tensor([0])
    in_dist_pred = torch.tensor([0])
    out_dist_labels = torch.tensor([1])
    out_dist_pred = torch.tensor([1])
    print("Case 5 - Small Tensor Sizes:")
    print(
        f"OOD Score (Accuracy): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='accuracy')}"
    )
    print(
        f"OOD Score (F1 Score): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='f1_score')}\n"
    )

    # Larger size
    in_dist_labels = torch.randint(0, 2, (1_000,))
    in_dist_pred = torch.randint(0, 2, (1_000,))
    out_dist_labels = torch.randint(0, 2, (1_000,))
    out_dist_pred = torch.randint(0, 2, (1_000,))
    print("Case 5 - Large Tensor Sizes:")
    print(
        f"OOD Score (Accuracy): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='accuracy')}"
    )
    print(
        f"OOD Score (F1 Score): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='f1_score')}\n"
    )

    # Case 6: Non-binary classes (multi-class scenario)
    in_dist_labels = torch.tensor([0, 1, 2, 2, 1])
    in_dist_pred = torch.tensor([0, 1, 2, 0, 1])  # Mix of correct and incorrect
    out_dist_labels = torch.tensor([0, 1, 2, 2, 1])
    out_dist_pred = torch.tensor([1, 2, 0, 1, 0])  # Mostly incorrect for OOD
    print("Case 6 - Non-Binary Classes (Multi-Class):")
    print(
        f"OOD Score (Accuracy): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='accuracy')}"
    )
    print(
        f"OOD Score (F1 Score): {ood_score_func(in_dist_pred, out_dist_pred, in_dist_labels, out_dist_labels, metric='f1_score')}\n"
    )

    # Case 7: Similar in-distribution and out-of-distribution
    in_dist_labels = torch.ones(1_000, dtype=torch.long)
    in_dist_pred = torch.zeros(1_000, dtype=torch.long)
    in_dist_pred[-1] = 1
    out_dist_labels = torch.ones(2_000, dtype=torch.long)
    out_dist_pred = torch.zeros(2_000, dtype=torch.long)
    out_dist_pred[-1] = 1
    print("Case 7 - Similar In-Distribution and Out-of-Distribution near zero:")
    print(
        f"OOD Score (Accuracy): {ood_score_func(in_dist_pred=in_dist_pred, out_dist_pred=out_dist_pred, in_dist_labels=in_dist_labels, out_dist_labels=out_dist_labels, metric='accuracy')}"
    )
    print(
        f"OOD Score (F1 Score): {ood_score_func(in_dist_pred=in_dist_pred, out_dist_pred=out_dist_pred, in_dist_labels=in_dist_labels, out_dist_labels=out_dist_labels, metric='f1_score')}\n"
    )

    # Case 8 Similar in-distribution and out-of-distribution
    in_dist_labels = torch.ones(2_000, dtype=torch.long)
    in_dist_pred = torch.zeros(2_000, dtype=torch.long)
    in_dist_pred[-1] = 1
    out_dist_labels = torch.ones(1_000, dtype=torch.long)
    out_dist_pred = torch.zeros(1_000, dtype=torch.long)
    out_dist_pred[-1] = 1
    print("Case 8 - Similar In-Distribution and Out-of-Distribution near zero:")
    print(
        f"OOD Score (Accuracy): {ood_score_func(in_dist_pred=in_dist_pred, out_dist_pred=out_dist_pred, in_dist_labels=in_dist_labels, out_dist_labels=out_dist_labels, metric='accuracy')}"
    )
    print(
        f"OOD Score (F1 Score): {ood_score_func(in_dist_pred=in_dist_pred, out_dist_pred=out_dist_pred, in_dist_labels=in_dist_labels, out_dist_labels=out_dist_labels, metric='f1_score')}\n"
    )


if __name__ == "__main__":
    test_ood_score()
