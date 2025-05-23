"""
DESCRIPTION
- first soruce for mahalanobis detector is: https://arxiv.org/abs/1807.03888 with the repostiory https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/lib_generation.py
- this implementation, is based on https://github.com/jfc43/robust-ood-detection/blob/master/CIFAR/eval_ood_detection.py#L426
"""
import logging

from tqdm import tqdm
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import sklearn.covariance
from sklearn.metrics import roc_curve
import mlflow

from ood_score import ood_score_func
from model.transformer_decoder import MyTransformerDecoder


def eval_mahalanobis(
    model,
    num_classes,
    sample_mean,
    precision,
    regressor,
    magnitude,
    testloader,
    threshold,
    quality_based,
    temperature,
    use_mlflow: bool = False
) -> None:
    """
    When the binary classifier for detecting ood is trained, evaluate the prediction on id and ood dataset with this function
    Args:
        model: a trained model for quality prediction
        num_classes (int): number of classes in the dataset (e.g. 2)
        sample_mean (list): contains the list with the mean activations for the different classes based on the training dataset
        precision (list): inverse of the covariance matrix for the classes
        regressor: trained binary classificator for ood prediction
        magnitude (float): parameter for apllying pertubations on input
        trainloader: loader for training data
        threshold (float): the threshold based on the val dataset
        quality_based (bool): if the binary classifier is based on quality labels (= TRUE) or based on predicting the models correct and incorrect predictions (=False)
        temperature (float): parameter for temperature calibration
        use_mlflow (bool): whether to use mlflow for logging
    """

    confidence_test = []
    confidence_train = []
    labels_test = []
    labels_train = []
    labels_quality_test_all = (
        []
    )  # save quality labels for calculating ood score when using testdata

    model.eval()

    # set information about feature extaction
    if model.use_latent_input:
        temp_x = torch.randint(0, model.num_latent_tokens, (2, model.input_size), device=model.device)
        temp_x = model.embedding(temp_x)
    else:
        temp_x = torch.rand(2, model.input_size * model.in_dim)
    temp_x = Variable(temp_x).to(model.device)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)


    count = 0
    is_transformer = isinstance(model, MyTransformerDecoder)
    for j, data in enumerate(tqdm(testloader, desc="Processing test (OOD) data")):
        if is_transformer:
            timeseries, labels_quality, _ = data
        else:
            timeseries, labels_quality = data

        labels_quality_test_all.extend(labels_quality.detach().cpu())

        if quality_based:
            labels = labels_quality
        else:
            labels = get_prediction_labels(model, timeseries, labels_quality)

        batch_size = timeseries.shape[0]

        inputs = timeseries

        mahalanobis_scores = get_Mahalanobis_score(
            model,
            inputs,
            num_classes,
            sample_mean,
            precision,
            num_output,
            magnitude,
            temperature,
        )

        confidence_scores = regressor.predict_proba(mahalanobis_scores)[:, 1]
        confidence_test.extend(confidence_scores)
        labels_test.extend(labels)

        count += batch_size


    logging.info("out of distribution dataset size: %d", len(confidence_test))
    threshold, _ = get_threshold_and_detect(
        labels=labels_test,
        confidences=confidence_test,
        dataset_name="test_dataset",
        threshold=threshold,
        calculate_ood_score=True,
        dataset_type="test",
        labels_quality=labels_quality_test_all,
        quality_based=quality_based,
        use_mlflow=use_mlflow
    )



def get_Mahalanobis_score(
    model, data, num_classes, sample_mean, precision, num_output, magnitude, temperature
):
    """
    Calculating the distance of the activations of a sample and the mean values of the classes, involves applying temperature scaling and pertubations for correcting the original prediction models softmax predictions,
    and  calculation of the gaussian score
      Args:
        model: a trained model for quality prediction
        num_classes (int): number of classes in the dataset (e.g. 2)
        sample_mean (list): contains the list with the mean activations for the different classes based on the training dataset
        precision (list): inverse of the covariance matrix for the classes
        num_putput (int): number of output classes
        magnitude (float): parameter for apllying pertubations on input
        temperature (int): parameter for calibrating models output (NOT USED)
    """

    for layer_index in range(num_output):

        if model.use_latent_input and data.dtype == torch.long:
            data = model.embedding(data.to(model.device))

        data = Variable(data, requires_grad=True).to(model.device)
        data.retain_grad()

        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # Collect scores for each class in a list
        gaussian_scores_list = []
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features - batch_sample_mean
            term_gau = (
                -0.5
                * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            )
            gaussian_scores_list.append(term_gau.view(-1, 1))

        # Concatenate scores after the loop
        gaussian_score = torch.cat(gaussian_scores_list, dim=1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = (
            -0.5
            * torch.mm(
                torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()
            ).diag()
        )
        loss = torch.mean(-pure_gau)
        loss.backward()

        assert data.grad is not None, "Gradient should exist after backward pass"
        gradient = torch.ge(data.grad, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(data, gradient, alpha=-magnitude)
        noise_out_features = model.intermediate_forward(
            Variable(tempInputs), layer_index
        )
        noise_out_features = noise_out_features.view(
            noise_out_features.size(0), noise_out_features.size(1), -1
        )
        noise_out_features = torch.mean(noise_out_features, 2)

        # Collect noise scores for each class in a list
        noise_gaussian_scores_list = []
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = (
                -0.5
                * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            )
            noise_gaussian_scores_list.append(term_gau.view(-1, 1))

        # Concatenate noise scores after the loop
        noise_gaussian_score = torch.cat(noise_gaussian_scores_list, dim=1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        noise_gaussian_score = np.asarray(
            noise_gaussian_score.cpu().numpy(), dtype=np.float32
        )
        if layer_index == 0:
            mahalanobis_scores = noise_gaussian_score.reshape(
                (noise_gaussian_score.shape[0], -1)
            )
        else:
            mahalanobis_scores = np.concatenate(
                (
                    mahalanobis_scores,
                    noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1)),
                ),
                axis=1,
            )

    return mahalanobis_scores


def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    Compute sample mean and precision (inverse of covariance).

    Returns:
        sample_class_mean (list[torch.Tensor]): List of class means per layer.
        precision (list[torch.Tensor]): List of precision matrices per layer.
    """
    device = model.device # Assuming model has a device attribute
    model.eval()
    # No need to move model to device here if it's already done outside

    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)

    # Simplified initialization
    num_sample_per_class = torch.zeros(num_classes, dtype=torch.long, device=device)

    # Initialize list to collect features [(layer)][(class)][features]
    # We'll store lists of tensors and concatenate later
    collected_features = [
        [[] for _ in range(num_classes)] for _ in range(num_output)
    ]
    is_transformer = isinstance(model, MyTransformerDecoder)
    with torch.no_grad(): # No need for gradients here
        for batch in tqdm(train_loader, desc="Estimating mean and covariance"):
            if is_transformer:
                data, target, _ = batch
            else:
                data, target = batch

            batch_size = data.size(0)
            total += batch_size
            data = data.to(device)
            target = target.to(device) # Ensure target is on the same device for comparison

            # Assuming model.feature_list returns output and list of features
            # Make sure features are detached if model is in eval mode and no_grad is used

            if model.use_latent_input:
                data = model.embedding(data)

            output, out_features = model.feature_list(data)

            # Process features for each layer
            processed_features = []
            for i in range(num_output):
                # Ensure features stay on the correct device
                feat = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                feat = torch.mean(feat, 2)
                processed_features.append(feat) # Keep features on device

            # Optional: compute training accuracy within this pass
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()

            # Accumulate features per class
            for i in range(batch_size):
                label = target[i].item() # Get scalar label index
                num_sample_per_class[label] += 1
                for layer_idx in range(num_output):
                    # Append tensor (view ensures it's 2D: 1 x num_features)
                    # Keep collected features on the original device
                    collected_features[layer_idx][label].append(
                        processed_features[layer_idx][i].view(1, -1)
                    )

    # Concatenate collected features after the loop
    # Now list_features[k][j] will be a tensor of features for layer k, class j
    list_features = [[None for _ in range(num_classes)] for _ in range(num_output)]
    for k in range(num_output):
        for j in range(num_classes):
            if collected_features[k][j]: # Check if list is not empty
                list_features[k][j] = torch.cat(collected_features[k][j], dim=0)
            else:
                logging.warning(
                    f"No samples found for layer {k}, class {j} during sample estimation."
                )
                # Handle appropriately - maybe skip class or raise error if unexpected

    # Calculate class means
    sample_class_mean = []
    for k in range(num_output):
        num_feature = int(feature_list[k])
        layer_means = torch.zeros(num_classes, num_feature, device=device)
        for j in range(num_classes):
            if list_features[k][j] is not None:
                layer_means[j] = torch.mean(list_features[k][j], dim=0)
            # else: mean remains zero, or handle as needed
        sample_class_mean.append(layer_means)

    # Calculate precision matrices
    precision = []
    for k in range(num_output):
        centered_features_list = []
        valid_classes_for_layer = 0
        for i in range(num_classes):
            if list_features[k][i] is not None:
                # Center features (ensure mean is also on the correct device)
                centered_features = list_features[k][i] - sample_class_mean[k][i]
                centered_features_list.append(centered_features)
                valid_classes_for_layer += 1

        if valid_classes_for_layer > 0 and centered_features_list:
             # Concatenate centered features for the layer
            X = torch.cat(centered_features_list, dim=0)

            # Fit covariance estimator (requires CPU numpy array)
            group_lasso.fit(X.detach().cpu().numpy())
            temp_precision = group_lasso.precision_
            # Convert back to tensor and move to original device
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            precision.append(temp_precision)
        else:
            logging.warning(f"Could not calculate precision for layer {k} due to missing class data.")
            # Append a placeholder or handle as needed. Appending None might cause issues later.
            # Let's append an identity matrix of appropriate size as a fallback,
            # assuming feature_list[k] gives the dimension
            num_feature = int(feature_list[k])
            precision.append(torch.eye(num_feature, device=device))


    logging.info(f"Sample Estimation Training Accuracy: ({100.0 * correct / total:.2f}%)")

    return sample_class_mean, precision


def balance_binary_dataset(inputs, labels):
    """
    For training the mahalanobis logistic regressor, a balanced dataset shoudl be used, no label overrepresented

    Args:
        inputs (np.ndarray): Input (Shape: [200, 2]).
        labels (np.ndarray): binary label (0 oder 1)

    Returns:
        dataset that contains the same amount of 1 and 0 samples
    """
    # 1 Extract indices and their number per class
    idx_class0 = np.where(labels == 0)[0]
    idx_class1 = np.where(labels == 1)[0]

    n_class0 = len(idx_class0)
    n_class1 = len(idx_class1)

    # 2 determine class with lower samples

    n_samples = min(n_class0, n_class1)

    # 3 Randomly select labels
    np.random.shuffle(idx_class0)
    np.random.shuffle(idx_class1)

    selected_idx = np.concatenate([idx_class0[:n_samples], idx_class1[:n_samples]])

    balanced_inputs = inputs[selected_idx]
    balanced_labels = labels[selected_idx]

    return balanced_inputs, balanced_labels


def get_prediction_labels(model, inputs, labels):
    """
    function to determine id/ood labels based on correct/incorrects prediction, then use these labels to train logistic regression to
    predict if model makes wrong prediction --> Assumption: at least part of the missclasssfied data is ood
    """
    with torch.no_grad():
        inputs = inputs.to(model.device)
        outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels.to(model.device)).int()
    return correct.cpu().numpy()


def _prepare_lr_data(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_size: int,
    val_size: int,
    quality_based: bool,
) -> tuple[list, list, list, list]:
    """
    Prepare training and validation data for the logistic regressor.

    Loads data up to specified sizes, extracts features or gets prediction
    labels based on `quality_based` flag.

    Args:
        model: The trained quality prediction model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        train_size: Maximum number of training samples to load.
        val_size: Maximum number of validation samples to load.
        quality_based: If True, use quality labels; otherwise, use
                       prediction correctness labels.

    Returns:
        A tuple containing:
        - train (list): List of training inputs (as numpy arrays).
        - val (list): List of validation inputs (as numpy arrays).
        - target_train (list): List of training targets.
        - target_val (list): List of validation targets.
    """
    train, val = [], []
    target_train, target_val = [], []

    # Process training data
    cnt = 0
    is_transformer = isinstance(model, MyTransformerDecoder)
    for batch in train_loader:
        if is_transformer:
            inputs, labels, _ = batch
        else:
            inputs, labels = batch
        train.extend(inputs.numpy())
        if quality_based:
            target_train.extend(labels.numpy())
        else:
            labels_prediction_based = get_prediction_labels(
                model, inputs, labels
            )
            target_train.extend(labels_prediction_based)
        cnt += len(inputs)
        if cnt >= train_size:
            break

    # Process validation data
    cnt = 0
    for batch in tqdm(
        val_loader, desc="Processing validation data for LR prep"
    ):
        if is_transformer:
            inputs, labels, _ = batch
        else:
            inputs, labels = batch
        val.extend(inputs.numpy())
        if quality_based:
            target_val.extend(labels.numpy())
        else:
            labels_prediction_based = get_prediction_labels(
                model, inputs, labels
            )
            target_val.extend(labels_prediction_based)
        cnt += len(inputs)
        if cnt >= val_size:
            break

    return train, val, target_train, target_val


def _calculate_mahalanobis_for_lr(
    model: torch.nn.Module,
    balanced_data: torch.Tensor,
    batch_size: int,
    num_classes: int,
    sample_mean: list[torch.Tensor],
    precision: list[torch.Tensor],
    num_output: int,
    magnitude: float,
    temperature: float,
) -> np.ndarray:
    """
    Calculate Mahalanobis scores for the balanced training data used
    for the logistic regression model.

    Args:
        model: The trained quality prediction model.
        balanced_data: The balanced input data tensor.
        batch_size: Batch size for processing.
        num_classes: Number of classes.
        sample_mean: List of class means per layer.
        precision: List of precision matrices per layer.
        num_output: Number of feature layers.
        magnitude: Perturbation magnitude.
        temperature: Temperature scaling parameter.

    Returns:
        An array of Mahalanobis scores for the input data.
    """
    train_lr_mahalanobis = []
    num_batches = (balanced_data.size(0) + batch_size - 1) // batch_size
    for batch_idx in tqdm(
        range(num_batches), desc="Calculating Mahalanobis scores for LR"
    ):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, balanced_data.size(0))
        data = balanced_data[start_idx:end_idx]

        mahalanobis_scores = get_Mahalanobis_score(
            model,
            data,
            num_classes,
            sample_mean,
            precision,
            num_output,
            magnitude,
            temperature,
        )
        train_lr_mahalanobis.extend(mahalanobis_scores)

    return np.asarray(train_lr_mahalanobis, dtype=np.float32)


def _get_validation_confidences(
    model: torch.nn.Module,
    regressor: LogisticRegressionCV,
    val_data: list,
    val_size: int,
    batch_size: int,
    num_classes: int,
    sample_mean: list[torch.Tensor],
    precision: list[torch.Tensor],
    num_output: int,
    magnitude: float,
    temperature: float,
) -> list[float]:
    """
    Calculate confidence scores for the validation set using the trained
    logistic regressor.

    Args:
        model: The trained quality prediction model.
        regressor: The trained logistic regression model.
        val_data: List of validation inputs (as numpy arrays).
        val_size: Size of the validation dataset to process.
        batch_size: Batch size for processing.
        num_classes: Number of classes.
        sample_mean: List of class means per layer.
        precision: List of precision matrices per layer.
        num_output: Number of feature layers.
        magnitude: Perturbation magnitude.
        temperature: Temperature scaling parameter.

    Returns:
        A list of confidence scores for the validation data.
    """
    confidence_val = []
    count = 0
    num_val_batches = (val_size + batch_size - 1) // batch_size
    for i in tqdm(range(num_val_batches), desc="Processing validation set"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, val_size)
        if start_idx >= val_size:
            break
        # Convert list slice to numpy array first
        images = torch.tensor(np.array(val_data)[start_idx:end_idx])
        current_batch_size = images.shape[0]

        mahalanobis_scores = get_Mahalanobis_score(
            model,
            images,
            num_classes,
            sample_mean,
            precision,
            num_output,
            magnitude,
            temperature,
        )

        confidence_scores = regressor.predict_proba(mahalanobis_scores)[:, 1]
        confidence_val.extend(confidence_scores)

        count += current_batch_size
    return confidence_val


def tune_mahalanobis_hyperparams(
    model,
    trainloader,
    valloader,
    num_classes,
    batch_size,
    train_size,
    val_size,
    quality_based,
    magnitude,
    temperature,
):
    """
    Function for calculating mean activations for each class, calculating mahalanobis scores and training a binary classificator on detecting ood based on the mahalanobis scores. After training, the threshold is set based on the validation data
    The mahalanobis based detecot can either be trained on quality labels or to predict missclassifications, choose one option by setting "quality_based" True or False
    Args:
        model: a trained model for quality prediction
        trainloader: loader for training data
        valloader: loader for validation data
        num_classes (int): number of classes in the dataset (e.g. 2)
        batch_size (int): batch size in the data
        val_size (int): size of the validation dataset
        quality_based (bool): if the binary classifier is based on quality labels (= TRUE) or based on predicting the models correct and incorrect predictions (=False)
        magnitude (float): parameter for apllying pertubations on input
        temperature (int): parameter for calibrating the models output softmax
    returns
        sample_mean (list), precision (list), best_regressor (model), magnitude (float), threshold (float)

    """
    logging.info("Tuning hyper-parameters...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)
    # 1 Calculation of class means of activations based on the train dataset
    # set information about feature extaction
    
    if model.use_latent_input:
        temp_x = torch.randint(0, model.num_latent_tokens, size=(2, model.input_size), device=model.device)
        temp_x = model.embedding(temp_x)
    else:
        temp_x = torch.rand(2, model.input_size * model.in_dim)
    temp_x = Variable(temp_x).to(device)
    temp_list = model.feature_list(temp_x)[1]

    temp_list = [out.cpu() for out in temp_list]

    num_output = len(temp_list)
    feature_list = np.empty(num_output)

    for count, out in enumerate(temp_list):
        feature_list[count] = out.size(1)

    logging.info("Get sample mean and covariance")
    sample_mean, precision = sample_estimator(
        model, num_classes, feature_list, trainloader
    )

    # 2 Train a regresson model either based on correctness of prediction or quality labels
    logging.info("Train logistic regression model")
    # Step 1: Prepare data for Logistic Regression
    train, val, target_train, target_val = _prepare_lr_data(
        model,
        trainloader,
        valloader,
        train_size,
        val_size,
        quality_based,
    )


    train_lr_data = torch.tensor(np.array(train))
    train_lr_label = torch.tensor(np.array(target_train))


    best_fpr = 1.1
    best_magnitude = 0.0 # This seems unused, maybe related to tuning magnitude?

    # Balance the dataset for LR training
    balanced_data, balanced_labels = balance_binary_dataset(
        train_lr_data.numpy(), train_lr_label.numpy() # balance_binary_dataset expects numpy
    )
    balanced_data = torch.tensor(balanced_data) # Convert back to tensor if needed by next step


    # Step 2: Calculate Mahalanobis scores for the balanced training data
    train_lr_mahalanobis = _calculate_mahalanobis_for_lr(
        model,
        balanced_data,
        batch_size,
        num_classes,
        sample_mean,
        precision,
        num_output,
        magnitude,
        temperature,
    )

    # Step 3: Train Logistic Regressor
    regressor = LogisticRegressionCV(cv=5).fit( # Added cv=5 for cross-validation
        train_lr_mahalanobis, balanced_labels
    )
    best_regressor = regressor # Assuming magnitude tuning isn't implemented yet
    logging.info(
        f"Logistic Regressor params: {regressor.coef_}, {regressor.intercept_}"
    )

    # Step 4: Process Validation Dataset to find threshold
    logging.info("Processing validation set to find threshold")
    confidence_val = _get_validation_confidences(
        model,
        regressor,
        val, # Pass the validation data list
        val_size,
        batch_size,
        num_classes,
        sample_mean,
        precision,
        num_output,
        magnitude,
        temperature,
    )

    # Step 5: Determine Threshold
    threshold, fpr = get_threshold_and_detect(
        labels=np.array(target_val), # Pass the validation targets
        confidences=np.array(confidence_val), # Pass the calculated confidences
        dataset_name="validation_dataset",
        threshold=False,
        calculate_ood_score=False,
        dataset_type="val",
        labels_quality=None, # Not needed for threshold calculation
        quality_based=quality_based,
    )

    # Assuming magnitude tuning is not implemented, we use the single magnitude
    if fpr is not None and fpr < best_fpr:
        best_fpr = fpr
        best_magnitude = magnitude # Store the magnitude used
        # best_regressor is already set

    logging.info(
        f"Best Logistic Regressor params: {best_regressor.coef_}, "
        f"{best_regressor.intercept_}"
    )
    logging.info(f"Best magnitude: {best_magnitude}") # Log the magnitude used

    return sample_mean, precision, best_regressor, best_magnitude, threshold



def get_threshold_and_detect(
    labels,
    confidences,
    dataset_name,
    threshold,
    calculate_ood_score,
    dataset_type,
    labels_quality,
    quality_based,
    use_mlflow: bool = False
):
    """
    function for constructing a roc curve with the id/ood predictions and labels, if the detetor was trained for quality prediction then at first Id and ood labels are obtained, otherwise labels are used directly for the curve

    Args:
        labels (np.ndarray): labels (either quality, or id/ood labels based on prediction errors)
        confidences (np.ndarray): confidences of the binary mahalanobis classificator
        dataset_name (string): val, train or test
        threshold: threshold for detection, either float or set to "False" if no threshold exists yet
        calculate_ood_score (bool): set "True" if calculation for ood score should be applied
        quality_based (bool): if

    Returns:
       Tuple[float,float] : threshold, fpr
    """
    if not threshold:
        fpr, tpr, thresholds_roc = roc_curve(labels, confidences)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds_roc[optimal_idx]

        # Detect ood based on threshold
        confidences = np.array(confidences)
        ood_predictions = confidences < optimal_threshold
        num_ood = sum(ood_predictions)
        percent_ood = sum(ood_predictions) / len(ood_predictions)
        logging.info(f"OOD predictions in validation set: {num_ood}")
        logging.info(f"Percentage of OOD in validation set: {percent_ood:.4f}")
        logging.info(f"Optimierter Schwellenwert basierend auf der ROC-Kurve: {optimal_threshold}")
        return optimal_threshold, fpr[optimal_idx]

    else:
        optimal_threshold = threshold
        confidences = np.array(confidences).flatten()
        ood_predictions = confidences < optimal_threshold
        num_ood = sum(ood_predictions)
        percent_ood = sum(ood_predictions) / len(ood_predictions)
        logging.info(f"OOD predictions in {dataset_name}: {num_ood}")
        logging.info(f"Percentage of OOD in {dataset_name}: {percent_ood:.4f}")
        if percent_ood == 0.0:
            logging.info("No OOD predictions found in the dataset.")
            return optimal_threshold, None

        if calculate_ood_score:
            pred_classes = (confidences > 0.5).astype(int)
            quality_predictions = torch.tensor(pred_classes)
            quality_labels = torch.tensor(labels_quality)

            # Aufteilen in OOD und ID
            predictions_ood = quality_predictions[ood_predictions]
            labels_ood = quality_labels[ood_predictions]

            predictions_id = quality_predictions[~ood_predictions]
            labels_id = quality_labels[~ood_predictions]
            score_f1 = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="f1_score")
            score_acc = ood_score_func(predictions_id, predictions_ood, labels_id, labels_ood, metric="accuracy")
            logging.info(f"OOD score for {dataset_type}: {score_f1:.2f} (F1), {score_acc:.2f} (Acc)")
            if use_mlflow:
                mlflow.log_metric(f"{dataset_type}/mahalanobis/ood_score/f1", score_f1)
                mlflow.log_metric(f"{dataset_type}/mahalanobis/ood_score/acc", score_acc)

        return optimal_threshold, None


def mahalanobis_detector(
    model,
    trainloader,
    valloader,
    num_classes,
    batch_size,
    testloader,
    train_size,
    val_size,
    quality_based,
    magnitude,
    temperature,
    use_mlflow: bool = False
):
    """
    function for (1) train a binary classificator based on mahalanobis scores to detect ood/id based on train data and and set a threshold for detecting ood based on val data and (2) evaluate on train and test data

    Args:
        model: trained quality prediction model
        trainloader: loader for training data
        valloader: loader for validation data
        num_classes: number of classes in the dataset
        batch_size: batch size
        testloader: loader for test data (ood)
        train_size: size (len) of the train dataset
        val_size: size (len) of the validation dataset
        quality_based (bool): set True if mahalanbis detetor should predict quality labels, or False if it should detect missclassifications of the quality predictor, either way the roc curve will be constructed based on missclassfied samples and the corresponding confidence of the detector
        magnitude (float): perutbations parameter
        temperature (int): scaling parameter
    """

    sample_mean, precision, best_regressor, magnitude, threshold = (
        tune_mahalanobis_hyperparams(
            model,
            trainloader,
            valloader,
            num_classes,
            batch_size,
            train_size,
            val_size,
            quality_based,
            magnitude,
            temperature,
        )
    )

    eval_mahalanobis(
        model,
        num_classes,
        sample_mean,
        precision,
        best_regressor,
        magnitude,
        testloader=testloader,
        threshold=threshold,
        quality_based=quality_based,
        temperature=temperature,
        use_mlflow=use_mlflow
    )
