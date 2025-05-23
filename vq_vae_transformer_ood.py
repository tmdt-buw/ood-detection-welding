import logging as log
from pathlib import Path
from tqdm import tqdm
import numpy as np
import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, f1_score as f1

from model.transformer_decoder import MyTransformerDecoder
from model.vq_vae_patch_embed import VQVAEPatch
from model.mlp import MLP
from data_loader.data_module import WeldingDataModule, SimpleDataModule
from data_loader.laten_ds_helper import create_latent_space_dataset_VQ_VAE_IDsOOD
from data_loader.datasets import MyLatentAutoregressiveDataset, ClassificationDataset
from utils import load_raw_data, load_val_test_idx
from ood_score import ood_score_func
from utils_ood import determine_ood_score_threshold, cross_entropy_loss


def compute_odd_score_for_model(
    model: MyTransformerDecoder | MLP,
    val_loader: DataLoader,
    test_loader: DataLoader,
    ood_values_val: np.ndarray,
    ood_values_test: np.ndarray,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    """
    Computes Out-of-Distribution (OOD) scores for a given classification
    model using pre-calculated OOD indicator values (e.g., VQ-VAE
    reconstruction error).

    It determines an OOD threshold based on the validation set's OOD
    indicator values and classification performance, then evaluates the
    model on the test set, splitting it into in-distribution (ID) and
    OOD samples based on the threshold. Finally, it reports OOD
    detection metrics (accuracy, F1) and overall classification
    performance.

    Args:
        model (MyTransformerDecoder | MLP): The classification model to
            evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        ood_values_val (np.ndarray): OOD indicator values for the
            validation set.
        ood_values_test (np.ndarray): OOD indicator values for the test
            set.
        device (str | torch.device): The device to run computations on.
            Defaults to "cpu".
    
    Returns:
        ood_score_accuracy (float): OOD score accuracy
        ood_score_f1 (float): OOD score F1 score
    """
    log.debug(
        f"OOD values val: {ood_values_val.shape} | OOD values test: {ood_values_test.shape}"
    )

    if len(ood_values_val) > 1 and ood_values_val.shape[1] > 1:
        ood_values_val = ood_values_val.mean(axis=1)
        ood_values_test = ood_values_test.mean(axis=1)

    ood_values_val = np.array(ood_values_val)
    ood_values_test = np.array(ood_values_test)

    val_preds = []
    val_labels = []
    with torch.no_grad():
        for x, y, _ in tqdm(val_loader, total=len(val_loader)):
            x = x.to(device)
            y = y.to(device)
            if isinstance(model, MyTransformerDecoder):
                logits = model(x, generate=False)
            else:
                logits = model(x)
            preds = F.log_softmax(logits, dim=1).argmax(dim=1)

            val_preds.append(preds)
            val_labels.append(y)

    val_preds = torch.cat(val_preds, dim=0).cpu().numpy()
    val_labels = torch.cat(val_labels, dim=0).cpu().numpy()

    log.debug("Determining OOD threshold")
    threshold = determine_ood_score_threshold(
        prediction_array=val_preds,
        label_array=val_labels,
        ood_determiner_value=ood_values_val,
        ood_func="error",
    )
    # old threshold 
    # threshold = np.mean(ood_values_val) + 2 * np.std(ood_values_val)

    log.info(f"OOD threshold: {threshold:.4f}")

    test_preds = []
    test_labels = []
    with torch.no_grad():
        for x, y, _ in tqdm(test_loader, total=len(test_loader)):
            x = x.to(device)
            y = y.to(device)
            if isinstance(model, MyTransformerDecoder):
                logits = model(x, generate=False)
            else:
                logits = model(x)
            preds = F.log_softmax(logits, dim=1).argmax(dim=1)

            test_preds.append(preds)
            test_labels.append(y)

    log.debug(
        f"Test loss: {np.mean(ood_values_test):.4f} | Test loss std: {np.std(ood_values_test):.4f}"
    )
    test_preds = torch.cat(test_preds, dim=0).cpu()
    test_labels = torch.cat(test_labels, dim=0).cpu()

    if ood_values_test.ndim > 1:
        ood_values_test = ood_values_test.squeeze(-1)

    in_dist_idx = ood_values_test <= threshold
    out_dist_idx = ood_values_test > threshold
    log.info(f"In-dist: {in_dist_idx.sum()} | Out-dist: {out_dist_idx.sum()} | In-dist: {in_dist_idx.sum() / len(in_dist_idx):.4f} | Out-dist: {out_dist_idx.sum() / len(out_dist_idx):.4f}")

    in_dist_preds = test_preds[in_dist_idx]
    in_dist_labels = test_labels[in_dist_idx]

    out_dist_preds = test_preds[out_dist_idx]
    out_dist_labels = test_labels[out_dist_idx]

    ood_score_accuracy = ood_score_func(
        in_dist_pred=in_dist_preds,
        out_dist_pred=out_dist_preds,
        in_dist_labels=in_dist_labels,
        out_dist_labels=out_dist_labels,
        metric="accuracy",
    )

    ood_score_f1 = ood_score_func(
        in_dist_pred=in_dist_preds,
        out_dist_pred=out_dist_preds,
        in_dist_labels=in_dist_labels,
        out_dist_labels=out_dist_labels,
        metric="f1_score",
    )

    log.info("--------------------------------")
    log.info(
        f"OOD score accuracy: {ood_score_accuracy:.4f} - [{len(in_dist_labels)}/{len(test_labels)}] | OOD score f1: {ood_score_f1:.4f} - [{len(out_dist_labels)}/{len(test_labels)}]"
    )
    accuracy_score = accuracy(preds=test_preds, target=test_labels, task="binary")
    f1_score = f1(preds=test_preds, target=test_labels, task="binary")
    log.info(f"Accuracy: {accuracy_score:.4f} | F1 Score: {f1_score:.4f}")
    return ood_score_accuracy, ood_score_f1


def get_recon_ood_values(
    vq_vae_model: VQVAEPatch,
    val_loader: DataLoader,
    test_loader: DataLoader,
    n_cycles: int = 1,
    device: str = "cpu",
    data_path: Path = Path("data/Welding"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieves or calculates OOD (Out-of-Distribution) values for validation and test data.
    
    This function either loads pre-computed OOD values from disk or computes them using
    the provided VQ-VAE model. The computed values include latent representations and 
    reconstruction/encoder losses that can be used for OOD detection.
    
    Args:
        vq_vae_model: The VQ-VAE model used to generate latent representations and losses.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        n_cycles: Number of cycles per data sample.
        device: Device to run computations on.
        data_path: Path to directory for saving/loading OOD values.
    
    Returns:
        A tuple containing eight arrays:
        - val_ids_x: Latent representations for validation data.
        - val_y: Labels for validation data.
        - val_ood_recon_loss: Reconstruction loss for validation data (OOD indicator).
        - val_ood_enc_loss: Encoder loss for validation data (OOD indicator).
        - test_ids_x: Latent representations for test data.
        - test_y: Labels for test data.
        - test_ood_recon_loss: Reconstruction loss for test data (OOD indicator).
        - test_ood_enc_loss: Encoder loss for test data (OOD indicator).
    """
    path_recon_ood_values = data_path / f"id_recon_ood_values_{n_cycles}"

    if not path_recon_ood_values.exists():
        log.info(f"Computing OOD values for VQ-VAE {n_cycles} cycles")
        val_ids_x, val_y, val_ood_recon_loss, val_ood_enc_loss = (
            create_latent_space_dataset_VQ_VAE_IDsOOD(
                latent_space_model=vq_vae_model,
                loader=val_loader,
                seq_len=n_cycles,
                device=device,
            )
        )
        path_recon_ood_values.mkdir(parents=True, exist_ok=True)
        np.save(path_recon_ood_values / "val_ood_recon_loss.npy", val_ood_recon_loss)
        np.save(path_recon_ood_values / "val_ood_enc_loss.npy", val_ood_enc_loss)
        np.save(path_recon_ood_values / "val_ids_x.npy", val_ids_x)
        np.save(path_recon_ood_values / "val_y.npy", val_y)

        test_ids_x, test_y, test_ood_recon_loss, test_ood_enc_loss = (
            create_latent_space_dataset_VQ_VAE_IDsOOD(
                latent_space_model=vq_vae_model,
                loader=test_loader,
                seq_len=n_cycles,
                device=device,
            )
        )
        np.save(path_recon_ood_values / "test_ood_recon_loss.npy", test_ood_recon_loss)
        np.save(path_recon_ood_values / "test_ood_enc_loss.npy", test_ood_enc_loss)
        np.save(path_recon_ood_values / "test_ids_x.npy", test_ids_x)
        np.save(path_recon_ood_values / "test_y.npy", test_y)
    else: 
        val_ids_x = np.load(path_recon_ood_values / "val_ids_x.npy")
        val_y = np.load(path_recon_ood_values / "val_y.npy")
        val_ood_recon_loss = np.load(path_recon_ood_values / "val_ood_recon_loss.npy")
        val_ood_enc_loss = np.load(path_recon_ood_values / "val_ood_enc_loss.npy")
        test_ids_x = np.load(path_recon_ood_values / "test_ids_x.npy")
        test_y = np.load(path_recon_ood_values / "test_y.npy")
        test_ood_recon_loss = np.load(path_recon_ood_values / "test_ood_recon_loss.npy")
        test_ood_enc_loss = np.load(path_recon_ood_values / "test_ood_enc_loss.npy")
        

    return val_ids_x, val_y, val_ood_recon_loss, val_ood_enc_loss, test_ids_x, test_y, test_ood_recon_loss, test_ood_enc_loss


def test_vq_vae_model(
    data_path: Path,
    vq_vae_model: VQVAEPatch,
    classification_model: MyTransformerDecoder | MLP,
    n_cycles: int = 1,
    device: str = "cpu",
    use_mlflow: bool = False,
) -> None:
    """
    Tests a VQ-VAE model in conjunction with a classification model
    (Transformer or MLP) for OOD detection.

    This function loads welding data, generates latent representations and
    OOD indicators (reconstruction and encoder loss) using the VQ-VAE
    model, prepares datasets for the classification model, and then calls
    `compute_odd_score_for_model` twice: once using reconstruction loss
    and once using encoder loss as the OOD indicator.

    Args:
        data_path (Path): Path to the directory containing the welding
            data.
        vq_vae_model (VQVAEPatch): The pre-trained VQ-VAE model.
        classification_model (MyTransformerDecoder | MLP): The pre-trained
            classification model (Transformer or MLP).
        n_cycles (int): The number of cycles per data sample. Defaults to
            1.
        device (str): The device to run computations on. Defaults to "cpu".
        use_mlflow (bool): Flag indicating whether to use MLflow for
            logging (currently unused). Defaults to False.
    """
    classification_model.eval()
    vq_vae_model.eval()

    log.info("Loading data for VQ-VAE")
    val_idx, test_idx = load_val_test_idx(data_path / "Welding")
    ds, labels, exp_ids = load_raw_data(data_path / "Welding")

    data_module = WeldingDataModule(
        ds,
        labels,
        exp_ids,
        batch_size=512,
        val_split_idx=val_idx,
        test_split_idx=test_idx,
        n_cycles=n_cycles,
        ds_type="classification",
    )

    data_module.setup()

    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    vq_vae_model = vq_vae_model.to(device)

    (
        val_ids_x, 
        val_y, 
        val_ood_recon_loss, 
        val_ood_enc_loss, 
        test_ids_x, 
        test_y, 
        test_ood_recon_loss, 
        test_ood_enc_loss
    ) = get_recon_ood_values(
        vq_vae_model=vq_vae_model,
        val_loader=val_loader,
        test_loader=test_loader,
        n_cycles=n_cycles,
        device=device,
    )
    

    log.debug(
        f"Val OOD recon loss: {val_ood_recon_loss.shape[0]} | Val OOD enc loss: {val_ood_enc_loss.shape[0]}"
    )
    log.debug(
        f"Test OOD recon loss: {test_ood_recon_loss.shape[0]} | Test OOD enc loss: {test_ood_enc_loss.shape[0]}"
    )

    val_ids_x = val_ids_x.reshape(-1, n_cycles * 16)
    test_ids_x = test_ids_x.reshape(-1, n_cycles * 16)


    val_ds = MyLatentAutoregressiveDataset(data=val_ids_x, y=val_y)
    test_ds = MyLatentAutoregressiveDataset(data=test_ids_x, y=test_y)

    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512)

    log.info("--------------------------------")
    log.info("Computing OOD score Reconstruction Error for VQ-VAE")
    ood_recon_accuracy, ood_recon_f1 = compute_odd_score_for_model(
        model=classification_model,
        val_loader=val_loader,
        test_loader=test_loader,
        ood_values_val=val_ood_recon_loss,
        ood_values_test=test_ood_recon_loss,
        device=device,
    )

    log.info("--------------------------------")
    log.info("Computing OOD score Encoder Error for VQ-VAE")
    ood_enc_accuracy, ood_enc_f1 = compute_odd_score_for_model(
        model=classification_model,
        val_loader=val_loader,
        test_loader=test_loader,
        ood_values_val=val_ood_enc_loss,
        ood_values_test=test_ood_enc_loss,
        device=device,
    )

    if use_mlflow:
        mlflow.log_metric("test/ood_recon/accuracy", ood_recon_accuracy)
        mlflow.log_metric("test/ood_recon/f1", ood_recon_f1)
        mlflow.log_metric("test/ood_enc/accuracy", ood_enc_accuracy)
        mlflow.log_metric("test/ood_enc/f1", ood_enc_f1)

    log.info("--------------------------------")

def compute_odd_score_for_autoregressive_model(
    model: MyTransformerDecoder,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
) -> tuple[float, float]:
    """
    Computes Out-of-Distribution (OOD) scores for an autoregressive
    transformer model using its own cross-entropy loss as the OOD
    indicator.

    Similar to `compute_odd_score_for_model`, it determines an OOD
    threshold based on the validation set's autoregressive loss and
    performance, then evaluates the model on the test set, splitting it
    into ID and OOD samples. It reports OOD detection metrics and
    overall classification performance.

    Args:
        model (MyTransformerDecoder): The autoregressive transformer
            model.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): The device to run computations on. Defaults to "cpu".
        use_mlflow (bool): Flag indicating whether to use MLflow for
            logging (currently unused). Defaults to False.

    Returns:
        ood_score_accuracy (float): OOD score accuracy
        ood_score_f1 (float): OOD score F1 score
    """
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

    test_loss = []
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            x, y, y_gen = batch
            x = x.to(device)
            y = y.to(device)
            y_gen = y_gen.to(device)
            # compute ood score
            logits_autoregressive = model(x, generate=True)
            # compute loss
            loss_per_item = cross_entropy_loss(
                logits_autoregressive,
                y_gen,
                ignore_index=logits_autoregressive.shape[2],
            )

            test_loss.append(loss_per_item.cpu().detach().numpy())

            # prediction
            logits = model(x, generate=False)
            preds = F.log_softmax(logits, dim=1).argmax(dim=1)

            test_preds.append(preds)
            test_labels.append(y)

    test_loss = np.concatenate(test_loss, axis=0)
    log.debug(
        f"Test loss: {np.mean(test_loss):.4f} | Test loss std: {np.std(test_loss):.4f}"
    )
    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)

    in_dist_idx = test_loss <= threshold
    out_dist_idx = test_loss > threshold

    in_dist_preds = test_preds[in_dist_idx]
    in_dist_labels = test_labels[in_dist_idx]

    out_dist_preds = test_preds[out_dist_idx]
    out_dist_labels = test_labels[out_dist_idx]

    ood_score_accuracy = ood_score_func(
        in_dist_pred=in_dist_preds,
        out_dist_pred=out_dist_preds,
        in_dist_labels=in_dist_labels,
        out_dist_labels=out_dist_labels,
        metric="accuracy",
    )

    ood_score_f1 = ood_score_func(
        in_dist_pred=in_dist_preds,
        out_dist_pred=out_dist_preds,
        in_dist_labels=in_dist_labels,
        out_dist_labels=out_dist_labels,
        metric="f1_score",
    )

    log.info(
        f"OOD score accuracy: {ood_score_accuracy:.4f} - [{len(in_dist_labels)}/{len(test_labels)}] | OOD score f1: {ood_score_f1:.4f} - [{len(out_dist_labels)}/{len(test_labels)}]"
    )
    accuracy_score = accuracy(preds=test_preds, target=test_labels, task="binary")
    f1_score = f1(preds=test_preds, target=test_labels, task="binary")
    log.debug(f"Accuracy: {accuracy_score:.4f} | F1 Score: {f1_score:.4f}")
    return ood_score_accuracy, ood_score_f1

def test_autoregressive_model(
    model: MyTransformerDecoder,
    data_module: SimpleDataModule,
    device: str = "cpu",
    use_mlflow: bool = False,
) -> None:
    """
    Tests an autoregressive transformer model for OOD detection using its
    own loss.

    This function sets the model to evaluation mode, retrieves validation
    and test dataloaders from the data module, and then calls
    `compute_odd_score_for_autoregressive_model` to perform the OOD
    evaluation.

    Args:
        model (MyTransformerDecoder): The pre-trained autoregressive
            transformer model.
        data_module (SimpleDataModule): The data module containing train,
            validation, and test dataloaders.
        device (str): The device to run computations on. Defaults to "cpu".
        use_mlflow (bool): Flag indicating whether to use MLflow for
            logging (currently unused). Defaults to False.
    """
    log.info("--------------------------------")
    log.info("Computing OOD score for Autoregressive Model")

    model.eval()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    ood_score_accuracy, ood_score_f1 = compute_odd_score_for_autoregressive_model(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    if use_mlflow:
        mlflow.log_metric("test/ood_autoregressive/accuracy", ood_score_accuracy)
        mlflow.log_metric("test/ood_autoregressive/f1", ood_score_f1)
    log.info("--------------------------------")
