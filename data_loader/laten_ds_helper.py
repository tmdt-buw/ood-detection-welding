import logging as log
import torch
import numpy as np
from typing import Literal
from model.vq_vae_patch_embed import VQVAEPatch
from data_loader.data_module import WeldingDataModule
from torch.utils.data import DataLoader
from data_loader.datasets import MyLatentAutoregressiveDataset, MyLatentClassificationDataset
from tqdm import tqdm


def print_training_input_shape(
    data_module: WeldingDataModule, ds_type: str | None = None, seq_len: int = 1
):
    """
    Prints the input shape and data type of the training data.

    Args:
        data_module: The data module object that provides the training data.
        ds_type (str, optional): The type of dataset. Defaults to None.
        seq_len (int, optional): The sequence length. Defaults to 1.
    """

    if ds_type is not None:
        data_module.setup(ds_type=ds_type, seq_len=seq_len)
    else:
        data_module.setup(stage="fit", seq_len=seq_len)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    log.info(
        f"Len train loader: {len(train_loader.dataset)} Len val loader: {len(val_loader.dataset)} Len test loader: {len(test_loader.dataset)}"
    )
    batch = next(iter(val_loader))

    if ds_type == "reconstruction":
        log.info(f"Input shape: {batch.shape} type: {batch.dtype}")
    else:
        for i in range(len(batch)):
            log.info(f"Input {i} shape: {batch[i].shape} type: {batch[i].dtype}")


def get_latent_space_IDs(
    latent_space_model: VQVAEPatch, x: torch.Tensor, has_patch_embed: bool = False
):
    """
    Retrieves the latent space IDs for a given input tensor.

    Args:
        latent_space_model (object): The latent space model.
        x (torch.Tensor): The input tensor.
        has_patch_embed (bool, optional): Indicates whether the input tensor has patch embeddings.
            Defaults to False.

    Returns:
        torch.Tensor: The latent space IDs.

    """
    if has_patch_embed:
        x = latent_space_model.patch_embed(x)
    else:
        x = x.permute(0, 2, 1)
    z_e = latent_space_model.encoder(x)
    _, z_q, _, _, min_encoding_indices = latent_space_model.vector_quantization(z_e)

    return min_encoding_indices

def get_latent_space_IDs_OOD(latent_space_model: VQVAEPatch, x: torch.Tensor):
    """
    Retrieves the latent space IDs for a given input tensor.

    Args:
        latent_space_model (object): The latent space model.
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The latent space IDs.

    """

    x = latent_space_model.patch_embed(x)
    z_e = latent_space_model.encoder(x)
    loss_OOD, _, indices, _  = latent_space_model.vector_quantization.forward_ood(z_e)
    return loss_OOD, indices

def create_latent_space_dataset_VQ_VAE_IDs(
    latent_space_model: VQVAEPatch,
    loader: DataLoader,
    seq_len: int,
    has_patch_embed: bool = False,
    no_labels: bool = False,
    window_size: int = 200,
    device: str = "cpu",
):
    """
    This function is sampling the latent space IDs for classification.

    Args:
        latent_space_model (VQVAEPatch): The VQ-VAE model.
        loader (DataLoader): The dataloader to be used.
        seq_len (int): The sequence length.
        has_patch_embed (bool, optional): Whether the model has patch embedding. Defaults to False.
        no_labels (bool, optional): Whether the dataset has labels. Defaults to False.
        device (str, optional): The device to use. Defaults to "cpu".

    Returns:
        new_ds_x (np.array): The latent space dataset (batch_size, seq_len, embedding_dim).
        new_ds_y (np.array): The labels of the dataset (batch_size,).
    """
    enc_out_len = int(latent_space_model.enc_out_len)

    len_ds = len(loader.dataset)

    new_ds_x = np.zeros((len_ds, seq_len, enc_out_len), dtype=int)
    new_ds_y = np.zeros((len_ds,))
    log.info(f"new_ds_x shape: {new_ds_x.shape} new_ds_y shape: {new_ds_y.shape}")

    latent_space_model.eval()
    batch_size = loader.batch_size
    with torch.no_grad():
        for b_i, batch_item in tqdm(enumerate(loader), total=len(loader)):
            if no_labels:
                x = batch_item
            else:
                x, y = batch_item

            for i in range(seq_len):
                x_i = x[:, i * window_size : (i + 1) * window_size, :].clone().detach()
                # log.info(f"shape: {x_i.shape}")
                x_i = x_i.to(device)
                ids = get_latent_space_IDs(
                    latent_space_model=latent_space_model,
                    x=x_i,
                    has_patch_embed=has_patch_embed,
                )
                ids = ids.cpu().numpy().reshape((x_i.shape[0], -1))
                new_ds_x[b_i * batch_size : (b_i + 1) * batch_size, i, :] = ids
            if not no_labels:
                new_ds_y[b_i * batch_size : (b_i + 1) * batch_size] = y.cpu().numpy()
    if no_labels:
        new_ds_y = np.zeros(new_ds_x.shape[0])
    new_ds_x = new_ds_x.reshape((new_ds_x.shape[0], -1))
    return new_ds_x, new_ds_y

def create_latent_space_dataset_VQ_VAE_IDsOOD(latent_space_model: VQVAEPatch, loader: DataLoader, seq_len: int, no_labels: bool=False, window_size: int = 200, device="cpu"):
    """
    This function is sampling the latent space IDs for classification.
    
    Args:
        latent_space_model (VQVAEPatch): The VQ-VAE model.
        loader (DataLoader): The dataloader to be used.
        seq_len (int): The sequence length.
        no_labels (bool, optional): Whether the dataset has labels. Defaults to False.
        device (str, optional): The device to use. Defaults to "cpu".
        
    Returns:
        new_ds_x (np.array): The latent space dataset (batch_size, seq_len, embedding_dim).
        new_ds_y (np.array): The labels of the dataset (batch_size,).
    """
    enc_out_len = int(latent_space_model.enc_out_len)

    len_ds = len(loader.dataset)

    new_ds_x = np.zeros((len_ds, seq_len, enc_out_len), dtype=int)
    new_ds_y = np.zeros((len_ds,))
    ood_recon_loss = np.zeros((len_ds,seq_len))
    ood_enc_loss = np.zeros((len_ds,seq_len))
    log.info(f"new_ds_x shape: {new_ds_x.shape} new_ds_y shape: {new_ds_y.shape}")

    latent_space_model.eval()
    batch_size = loader.batch_size
    with torch.no_grad():
        for b_i, batch_item in tqdm(enumerate(loader), total=len(loader)):
            if no_labels:
                x = batch_item
            else:
                x, y = batch_item
            
            for i in range(seq_len):
                x_i = x[:, i*window_size:(i+1)*window_size, :].clone().detach()
                # log.info(f"shape: {x_i.shape}")
                x_i = x_i.to(device)
                ood_loss, ids = get_latent_space_IDs_OOD(latent_space_model=latent_space_model, x=x_i)
                # log.info(f"ids shape: {ids.shape} ood_loss shape: {ood_loss.shape}")
                ids = ids.cpu().numpy().reshape((x_i.shape[0], -1))
                new_ds_x[b_i*batch_size:(b_i + 1) * batch_size, i, :] = ids

                _, x_hat, _ = latent_space_model(x_i)

                recon_error = torch.mean((x_i - x_hat) ** 2, dim=(1, 2))
                
                ood_recon_loss[b_i*batch_size:(b_i + 1) * batch_size, i] = recon_error.cpu().numpy()
                ood_enc_loss[b_i*batch_size:(b_i + 1) * batch_size, i] = ood_loss.cpu().numpy()
                # log.info(f"ids: {new_ds_x[b_i*batch_size, i, :]} {ids[0]}")
           
            if not no_labels:
                new_ds_y[b_i*batch_size:(b_i + 1) * batch_size] = y.cpu().numpy()
    if no_labels:
        new_ds_y = np.zeros(new_ds_x.shape[0])
    return new_ds_x, new_ds_y, ood_recon_loss, ood_enc_loss


def create_autoreg_ds(
    vq_vae_model: VQVAEPatch,
    data_module: WeldingDataModule,
    task: Literal["reconstruction", "classification"] = "reconstruction",
    seq_len: int = 1,
    device: str = "cpu",
    prob_unk_token: float = 0.0,
    seq_prediction_task: bool = True
) -> tuple[
    MyLatentAutoregressiveDataset | MyLatentClassificationDataset, 
    MyLatentAutoregressiveDataset | MyLatentClassificationDataset, 
    MyLatentAutoregressiveDataset | MyLatentClassificationDataset
]:
    """
    Creates autoregressive datasets for training, validation, and testing.

    Args:
        vq_vae_model: The VQ-VAE model.
        data_module: The data module object that provides the data loaders.
        task (str, optional): The type of autoregressive task. Defaults to "reconstruction".
        device (str, optional): The device to use. Defaults to "cpu".
        seq_prediction_task (bool, optional): Whether the task is a sequence prediction task. Defaults to True.
    Returns:
        tuple: A tuple containing the autoregressive datasets for training, validation, and testing.

    Raises:
        None

    Examples:
        # Create autoregressive datasets for reconstruction task
        train_ds, val_ds, test_ds = create_autoreg_ds(data_module, task="reconstruction")

        # Create autoregressive datasets for classification task
        train_ds, val_ds, test_ds = create_autoreg_ds(data_module, task="classification")
    """

    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    vq_vae_model = vq_vae_model.to(device)

    x_train, y_train = create_latent_space_dataset_VQ_VAE_IDs(
        vq_vae_model,
        loader=train_loader,
        seq_len=seq_len,
        has_patch_embed=True,
        no_labels=task == "reconstruction",
        device=device,
    )
    x_val, y_val = create_latent_space_dataset_VQ_VAE_IDs(
        vq_vae_model,
        loader=val_loader,
        seq_len=seq_len,
        has_patch_embed=True,
        no_labels=task == "reconstruction",
        device=device,
    )
    x_test, y_test = create_latent_space_dataset_VQ_VAE_IDs(
        vq_vae_model,
        loader=test_loader,
        seq_len=seq_len,
        has_patch_embed=True,
        no_labels=task == "reconstruction",
        device=device,
    )

    if seq_prediction_task:
        train_ds = MyLatentAutoregressiveDataset(x_train, y_train, prob_missing=prob_unk_token)
        val_ds = MyLatentAutoregressiveDataset(x_val, y_val, prob_missing=0.0)
        test_ds = MyLatentAutoregressiveDataset(x_test, y_test, prob_missing=0.0)
    else:
        train_ds = MyLatentClassificationDataset(x_train, y_train, prob_missing=prob_unk_token)
        val_ds = MyLatentClassificationDataset(x_val, y_val, prob_missing=0.0)
        test_ds = MyLatentClassificationDataset(x_test, y_test, prob_missing=0.0)

    return train_ds, val_ds, test_ds
