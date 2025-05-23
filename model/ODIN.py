"""
Sourrce:
CODE From original Paper https://github.com/facebookresearch/odin
Code here oriented on  https://github.com/jfc43/robust-ood-detection/blob/master/CIFAR/eval_ood_detection.py#L426

EXAMPLE:

epsilon = 0.0014 #For pertubation #Alternativley also called "magnitude"
temperature = 1000 #For scaling
threshold = ODIN(model, data_module.val_dataloader(), "val", temperature, epsilon, calculate_ood_score = False,) 
_ = ODIN(model, data_module.train_dataloader(), "train", temperature, epsilon, calculate_ood_score=False, threshold=threshold) 
_ = ODIN(model, data_module.test_dataloader(), "test", temperature, epsilon, calculate_ood_score=True, threshold=threshold) 

"""

import logging
import torch
import torch.nn as nn
import numpy as np
import mlflow
from sklearn.metrics import roc_curve
from ood_score import ood_score_func
from model.transformer_decoder import MyTransformerDecoder


def odin_softmax(model, data_loader, temper, noiseMagnitude1):
    """
    Decisive function that differes odin from MSP: get models predictions and inputs, but (1) apply temperature scaling on the predictions  and (2) pertubate the inputs most important parts (according to the gradients) to enhance difference between ID and OOD data
    and (3) get the adjusted softmax predictions by applying the model on the pertubated input

    Args:
        model: a trained model for quality prediction
        data_loader: dataloader for train, val or testdataset
        temper (int): temperature scaling parameter
        noiseMagnitude1(float): magnitude for pertubations

    Returns:
        Tuple: [adjusted softmax predictions, test labels]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    all_perturbed_outputs = []
    all_targets = []
    is_transformer = isinstance(model, MyTransformerDecoder)
    with torch.set_grad_enabled(True):  # Ensure gradients are enabled even in eval mode
        for batch in data_loader:
            if is_transformer:
                inputs_temp, targets, _ = batch
            else:
                inputs_temp, targets = batch
            inputs_temp = inputs_temp.to(device)
            targets = targets.to(device)
            
            if model.use_latent_input and inputs_temp.dtype == torch.long:
                inputs_temp = model.embedding(inputs_temp)
                if not is_transformer:
                    inputs_temp = inputs_temp.reshape(inputs_temp.shape[0], -1)                
                # Enable gradients for inputs
                inputs_temp.requires_grad_(True)
                inputs_temp.retain_grad()

                outputs = model.forward_layers(inputs_temp)
            else:

                # Enable gradients for inputs
                inputs_temp.requires_grad_(True)
                
                # Forward pass
                outputs = model(inputs_temp)
            
            # Get predicted labels
            max_index_temp = torch.argmax(outputs, dim=1)
            criterion = nn.CrossEntropyLoss()
            labels = max_index_temp
            
            # Apply temperature scaling
            scaled_outputs = outputs / temper
            
            # Compute loss and gradients
            loss = criterion(scaled_outputs, labels)
            loss.backward()
            
            # Create perturbation
            gradient = torch.ge(inputs_temp.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            if model.use_latent_input:
                norm = torch.norm(gradient, p=2, dim=1, keepdim=True)
            else:
                norm = torch.norm(gradient, p=2, dim=(1, 2), keepdim=True)
            gradient = gradient / norm
            perturbed_inputs = torch.add(
                inputs_temp.data, gradient, alpha=-noiseMagnitude1
            )
            
            # Get new predictions with perturbed inputs
            with torch.no_grad():
                if model.use_latent_input:
                    outputs_perturbed = model.forward_layers(perturbed_inputs)
                else:
                    outputs_perturbed = model(perturbed_inputs)
                outputs_perturbed = outputs_perturbed / temper
                softmax_probs = torch.nn.functional.softmax(outputs_perturbed, dim=1)
                
                all_perturbed_outputs.append(softmax_probs)
                all_targets.append(targets)
            
            # Clear gradients for next batch
            inputs_temp.grad.zero_()
    
    # Concatenate results
    all_perturbed_outputs = torch.cat(all_perturbed_outputs, dim=0).cpu()
    all_targets = torch.cat(all_targets, dim=0).cpu()
    
    return all_perturbed_outputs, all_targets


def right_wrong_fn(test_prediction, targets):
    """
    Function to check if the predictions of the model are actually right

    Args:
        test_prediction (np.ndarray): softmax predictions of the model
        targets (np.ndarray): real quality labels

    Returns:
        Tuple[np.ndarray, np.ndarray]: right, conf
    """
    # Calculate actual prediction
    pred_classes = np.argmax(test_prediction, axis=1)
    right = pred_classes == targets

    # confidence for the prediction
    conf = np.max(test_prediction, axis=1)

    return right, conf


def odin_ood_detector(test_prediction, Y_test, dataset_type, calculate_ood_score, threshold=None, use_mlflow: bool = False):
    """
    main function for detecting ood data, first deterimnes if predictions were right or wrong, and then uses that as labels for the roc curve based ood threshold definition

    Args:
        test_prediction (np.ndarray): softmax predictions of the model
        Y_test (np.ndarray): real quality labels
        dataset_type (string): information about what dataset is used (e.g. "val")
        threshold (float): optional give a already calculated threshold
        use_mlflow (bool): whether to use mlflow for logging
    Returns:
        fload: optimal_threshold
    """

    conf_right, conf_wrong, conf_all = [], [], []

    # 1 check if predictions "test_prediction" were right or wrong
    r, conf_a = right_wrong_fn(test_prediction, Y_test)
    r = np.array(r, dtype=bool)
    conf_all.extend(conf_a)
    conf_right.extend(conf_a[r])  # confidences for only correct predcitions
    conf_wrong.extend(conf_a[~r])  # conifdences for the incorrect predictions

    labels = np.array(
        [1] * len(conf_right) + [0] * len(conf_wrong)
    )  # 1 = In-Distribution, 0 = artificial "OOD" (prediction failures of the model)
    confidences = np.array(conf_right + conf_wrong)  # Softmax-Werte

    # 2 construct rock curve so a optimum point can be found where the model has a minimum amount of false classified  samples, defined by the maximum tpr/fpr difference
    fpr, tpr, thresholds_roc = roc_curve(labels, confidences)

    if threshold is None:
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds_roc[optimal_idx]
    else:
        optimal_threshold = threshold

    logging.info(f"ODIN threshold for {dataset_type}: {optimal_threshold}")

    # 3 OOD Detection based on optimal thresold
    ood_predictions = confidences < optimal_threshold
    dataset_size = len(confidences)
    logging.info(f"ODIN OOD samples in {dataset_type}: {sum(ood_predictions)}")


    if calculate_ood_score:
        pred_classes = np.argmax(test_prediction, axis=1)
        quality_predictions = torch.tensor(pred_classes)
        quality_labels = torch.tensor(Y_test)

        # Aufteilen in OOD und ID
        predictions_ood = quality_predictions[ood_predictions]
        labels_ood = quality_labels[ood_predictions]

        predictions_id = quality_predictions[~ood_predictions]
        labels_id = quality_labels[~ood_predictions]
        score_f1 = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="f1_score")
        score_acc = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="accuracy")
        logging.info(f"OOD score for {dataset_type}: {score_f1:.2f} (F1), {score_acc:.2f} (Acc)")
        if use_mlflow:
            mlflow.log_metric(f"{dataset_type}/odin/ood_score/f1", score_f1)
            mlflow.log_metric(f"{dataset_type}/odin/ood_score/acc", score_acc)

    return optimal_threshold


def ODIN(model, data_loader, dataset_type, temperature, pertubations, calculate_ood_score, threshold=None, use_mlflow: bool = False):
    """
    function for getting the adjusted softmax probabilites via temperature scaling and pertubations, and then perform threshold calculation identically to MSP
    Args:
        model: a trained model for quality prediction
        data_loader: dataloader for either the train,val or testset
        dataset_type (string): information about what dataset is used (e.g. "val")
        temperature (int): parameter for scaling
        pertubations (float): magnitude parameter for applying pertubations on input 
        calculate_ood_score (bool): set True to calculate ood score for the given dataset
        threshold (float): optional give a already calculated threshold
        use_mlflow (bool): whether to use mlflow for logging
    Returns:
        float: used threshold for detecting ood
    """
    softmax_probs, Y_test = odin_softmax(model, data_loader, temperature, pertubations)

    threshold = odin_ood_detector(
        test_prediction=softmax_probs.detach().numpy(),
        Y_test=Y_test.numpy(),
        dataset_type=dataset_type,
        calculate_ood_score=calculate_ood_score,
        threshold=threshold,
        use_mlflow=use_mlflow
    )
    return threshold
