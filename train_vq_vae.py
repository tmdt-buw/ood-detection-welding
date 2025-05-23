import logging as log
import argparse
from pathlib import Path
import numpy as np
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer

from model.vq_vae_patch_embed import VQVAEPatch
from data_loader.utils import load_raw_data
from data_loader.data_module import WeldingDataModule
from utils import get_logger, load_val_test_idx


def init_model(conf: dict[str, any]):
    log.info("Initializing VQ-VAE model with the following configuration:")
    log.info(conf)

    vq_vae_model = VQVAEPatch(
        hidden_dim=conf["hidden_dim"],
        input_dim=2,
        num_embeddings=conf["num_embeddings"],
        embedding_dim=conf["embedding_dim"],
        n_resblocks=conf["n_resblocks"],
        patch_size=25,
        seq_len=conf["seq_len"],
        kmeans_iters=conf.get("kmeans_iters", 5),
        threshold_ema_dead_code=conf.get("threshold_ema_dead_code", 2),
        batch_norm=conf.get("batch_norm", False),
        learning_rate=conf["learning_rate"],
    )

    return vq_vae_model


def train_vq_vae(data_module, conf: dict[str, any]):
    score = "val/loss"
    mode = "min"

    early_stop_callback = EarlyStopping(
        monitor=score, min_delta=0.001, patience=10, verbose=False, mode=mode
    )
    model_checkpoint = ModelCheckpoint(
        monitor=score, dirpath="model_checkpoints/VQ-VAE", filename="vq_vae", mode=mode
    )
    
    mlflow_logger = get_logger(use_mlflow=conf["use_mlflow"], experiment_name="ood-welding-vq-vae")

    vq_vae_model = init_model(conf)

    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=mlflow_logger,
        callbacks=[early_stop_callback, model_checkpoint],
        max_epochs=conf["epochs"],
        gradient_clip_val=conf["gradient_clip_val"],
    )

    log.info("Starting training")
    trainer.fit(vq_vae_model, data_module)
    best_val_loss = vq_vae_model.best_val_loss
    print(f"Best val loss: {best_val_loss}")
    log.info("Training finished")
    trainer.test(vq_vae_model, dataloaders=data_module.test_dataloader())
    return best_val_loss


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--hidden-dim", type=int, default=256)
    args.add_argument("--batch-size", type=int, default=512)
    args.add_argument("--patch-size", type=int, default=25)
    args.add_argument("--num-embeddings", type=int, default=256)
    args.add_argument("--embedding-dim", type=int, default=32)
    args.add_argument("--n-resblocks", type=int, default=4)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--learning-rate", type=float, default=1e-3)
    args.add_argument("--gradient-clip-val", type=float, default=1.0)
    args.add_argument("--dropout-p", type=float, default=0.1)
    args.add_argument("--seq-len", type=int, default=200)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--use-mlflow", action=argparse.BooleanOptionalAction)
    args.add_argument("--create-idx-files", action=argparse.BooleanOptionalAction)
    conf = vars(args.parse_args())
    log.info(conf)

    torch.set_float32_matmul_precision("medium")

    # lightning seed everthing
    pl.seed_everything(conf["seed"])

    data_path = Path("data/Welding/")

    # params
    if conf["create_idx_files"]:
        val_idx = None
        test_idx = None
    else:
        val_idx, test_idx = load_val_test_idx(data_path)

    log.info("Loading data")
    ds, labels, exp_ids = load_raw_data(data_path)

    data_module = WeldingDataModule(
        ds=ds,
        labels=labels,
        exp_ids=exp_ids,
        batch_size=conf["batch_size"],
        val_split_idx=val_idx,
        test_split_idx=test_idx,
        n_cycles=1,
        ds_type="reconstruction"
    )

    data_module.setup()

    if conf["create_idx_files"]:
        np.save(data_path / "val_idx.npy", data_module.val_idx)
        np.save(data_path / "test_idx.npy", data_module.test_idx)

    log.info("Initialize Training VQ-VAE")
    train_vq_vae(data_module, conf)


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
