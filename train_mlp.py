import argparse
from pathlib import Path
import logging as log
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer
import mlflow

from model.mlp import MLP
from data_loader.data_module import WeldingDataModule, SimpleDataModule
from utils import load_raw_data, get_logger, load_val_test_idx, get_laten_ds, load_first_stage_model
from post_hoc_ood import test_ood_detection
from vq_vae_transformer_ood import test_vq_vae_model


def init_model(conf: dict[str, any]) -> MLP:
    """Initialize the MLP model based on the provided configuration.

    Args:
        conf (dict[str, any]): A dictionary containing the model
            configuration parameters. Expected keys include:
            - n_cycles (int): Number of cycles to consider as input.
            - hidden_dim (int): The size of the hidden layers.
            - n_hidden_layers (int): The number of hidden layers.
            - dropout_p (float): The dropout probability.
            - learning_rate (float): The learning rate for the optimizer.
            - epochs (int): The total number of training epochs.
            - annealing_start (float): The starting value for annealing.

    Returns:
        MLP: An initialized MLP model instance.
    """
    log.info("Initializing MLP model with the following configuration:")
    log.info(conf)

    
    if conf["use_latent_input"]:
        num_latent_tokens = conf["num_latent_tokens"]
        input_size = int(conf["n_cycles"] * 2 * 8) + 1
    else:
        input_size = 200 * conf["n_cycles"]
        num_latent_tokens = 0

    mlp_model = MLP(
        input_size=input_size,
        output_size=2,
        in_dim=2,
        hidden_sizes=conf["hidden_dim"],
        n_hidden_layers=conf["n_hidden_layers"],
        dropout_p=conf["dropout_p"],
        learning_rate=conf["learning_rate"],
        annealing_step=conf["epochs"],
        annealing_start=conf["annealing_start"],
        use_edl_loss=conf["use_edl_loss"],
        use_latent_input=conf["use_latent_input"],
        num_latent_tokens=num_latent_tokens,
        use_layer_norm=conf["use_layer_norm"],
    )

    return mlp_model


def train_model(data_module, conf: dict[str, any]):
    """Train the MLP model using the provided data module and configuration.

    Steps:
    1. Define monitoring score and mode.
    2. Initialize EarlyStopping and ModelCheckpoint callbacks.
    3. Get the MLflow logger.
    4. Initialize the MLP model using `init_model`.
    5. Initialize the PyTorch Lightning Trainer.
    6. Start the training process using `trainer.fit`.

    Args:
        data_module: The data module providing training and validation
            dataloaders.
        conf (dict[str, any]): A dictionary containing the training
            configuration parameters. Expected keys include:
            - epochs (int): The total number of training epochs.
            - gradient_clip_val (float): Value for gradient clipping.
            - Other keys required by `init_model`.
    """
    score = "val/f1_score"
    mode = "max"

    early_stop_callback = EarlyStopping(
        monitor=score, min_delta=0.001, patience=15, verbose=False, mode=mode
    )
    model_checkpoint = ModelCheckpoint(
        monitor=score,
        dirpath=f"model_checkpoints/{conf['model_name']}",
        filename="mlp",
        save_top_k=1,
        save_last=True,
        mode=mode,
    )
    mlflow_logger = get_logger(conf["use_mlflow"])
    if conf["use_mlflow"]:
        mlflow_logger.log_hyperparams(
            {
                "gradient_clip_val": conf["gradient_clip_val"],
                "model_name": conf["model_name"],
                "batch_size": conf["batch_size"],
                "epochs": conf["epochs"],
                "seed": conf["seed"],
                "path_vq_vae": str(conf["path_vq_vae"]),
                "prob_unk_token": conf["prob_unk_token"],
                "n_cycles": conf["n_cycles"],
            }
        )

    model = init_model(conf)

    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=mlflow_logger,
        callbacks=[early_stop_callback, model_checkpoint],
        max_epochs=conf["epochs"],
        gradient_clip_val=conf["gradient_clip_val"],
    )

    log.info("Starting training")
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    best_val_loss = model.best_val_score
    print(f"Best val loss: {best_val_loss}")
    log.info("Training finished")

    test_model(model, data_module, mlflow_logger, conf)


def test_model(model, data_module, logger, conf: dict[str, any]):
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=logger,
    )
    trainer.test(model, dataloaders=data_module.test_dataloader())

    if conf["compute_ood_score"]:
        if conf["use_mlflow"]:
            if logger.run_id:
                run_id = logger.run_id
                log_context = mlflow.start_run(run_id=run_id)
                log_context.__enter__()  # Manually enter the context
            else:
                logger.my_logger.start_run()
                log_context.__enter__()  # Manually enter the context
            try:
                logger.log_hyperparams(
                    {
                        "seed": conf["seed"],
                        "model_name": conf["model_name"],
                    }
                )
            except Exception as e:
                log.error(f"Error logging hyperparams: {e}")
        try:
            test_ood_detection(model, data_module, conf["use_mlflow"])
            if conf["discret_model"] != "":
                device = "cuda"
                vq_vae_model = load_first_stage_model(conf["path_vq_vae"])
                vq_vae_model.to(device)
                test_vq_vae_model(
                    data_path=Path(conf["data_path"]),
                    vq_vae_model=vq_vae_model,
                    classification_model=model.to(device),
                    n_cycles=conf["n_cycles"],
                    device=device,
                    use_mlflow=conf["use_mlflow"],
                )
        finally:
            if conf["use_mlflow"]:
                log_context.__exit__(None, None, None)  # Manually exit the context

def get_data_module(conf: dict[str, any], data_path: Path):
    val_idx, test_idx = load_val_test_idx(data_path)

    log.info("Loading data")
    ds, labels, exp_ids = load_raw_data(data_path)

    if conf["discret_model"] == "":
        data_module = WeldingDataModule(
            ds,
            labels,
            exp_ids,
            batch_size=conf["batch_size"],
            val_split_idx=val_idx,
            test_split_idx=test_idx,
            n_cycles=conf["n_cycles"],
            ds_type="classification",
        )
        data_module.setup()
    elif conf["discret_model"] == "VQ-VAE":
        (
            recon_train_ds,
            recon_val_ds,
            recon_test_ds,
            class_train_ds,
            class_val_ds,
            class_test_ds,
        ) = get_laten_ds(
            vq_vae_path=Path(conf["path_vq_vae"]),
            init_ds=False,
            data_path=Path(conf["data_path"]),
            n_cycles=conf["n_cycles"],
            prob_unk_token=conf["prob_unk_token"],
            seq_prediction_task=False
        )

        data_module = SimpleDataModule(
            train_ds=class_train_ds,
            val_ds=class_val_ds,
            test_ds=class_test_ds,
            batch_size=conf["batch_size"],
        )
        data_module.setup()
        conf["num_latent_tokens"] = recon_train_ds.num_classes
    else:
        raise ValueError(f"Invalid discret model: {conf['discret_model']}")
    return data_module

def main():
    """Main function to set up and run the MLP model training process.

    Steps:
    1. Define the model and training configuration using argparse.
    2. Set the PyTorch float32 matmul precision.
    3. Seed everything using PyTorch Lightning for reproducibility.
    4. Define the data path and load validation/test indices.
    5. Load the raw data (datasets, labels, experiment IDs).
    6. Initialize the WeldingDataModule.
    7. Set up the data module for classification.
    8. Initialize and start the training process using `train_model`.
    """
    parser = argparse.ArgumentParser(description="Train an MLP model for welding quality prediction.")
    parser.add_argument("--discret-model", type=str, default="", help="VQ-VAE or EMPTY")
    parser.add_argument("--n-cycles", type=int, default=1, help="Number of cycles to consider as input.")
    parser.add_argument("--epochs", type=int, default=2, help="Total number of training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Size of the hidden layers.")
    parser.add_argument("--path-vq-vae", type=str, default="model_checkpoints/best_models/VQ-VAE/best_vq_vae.ckpt", help="Path to the discret model.")
    parser.add_argument("--n-hidden-layers", type=int, default=2, help="Number of hidden layers.")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for the optimizer.")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="Value for gradient clipping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--annealing-start", type=float, default=0.002, help="Starting value for annealing.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training and validation.")
    parser.add_argument("--data-path", type=str, default="data", help="Path to the data directory.")
    parser.add_argument("--prob-unk-token", type=float, default=0.0, help="Probability of unknown token.")
    parser.add_argument("--model-checkpoint-path", type=str, default="model_checkpoints/best_models/MLP/mlp-v56.ckpt", help="Path to the model checkpoint.")
    parser.add_argument("--use-mlflow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-edl-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-layer-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compute-ood-score", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test-only", action=argparse.BooleanOptionalAction, default=False)
    conf = vars(parser.parse_args())

    if conf["discret_model"] != "":
        conf["model_name"] = conf["discret_model"] + "_MLP"
        conf["use_latent_input"] = True
    else:
        if conf["use_edl_loss"]:
            conf["model_name"] = "MLP_EDL"
        else:
            conf["model_name"] = "MLP"
        conf["use_latent_input"] = False

        
    log.info(f"Configuration: {conf}")

    torch.set_float32_matmul_precision("medium")

    # lightning seed everthing
    pl.seed_everything(conf["seed"])

    data_path = Path(conf["data_path"]) / "Welding"

    data_module = get_data_module(conf, data_path)

    log.info("Initialize Training MLP")
    if conf["test_only"]:
        log.info("Testing MLP")
        model = MLP.load_from_checkpoint(conf["model_checkpoint_path"])
        mlflow_logger = get_logger(conf["use_mlflow"])
        test_model(model, data_module, mlflow_logger, conf)
    else:
        train_model(data_module, conf)


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
