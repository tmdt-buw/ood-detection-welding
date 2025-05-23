import logging as log
import argparse
from pathlib import Path
import torch
import mlflow
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import seed_everything
from lightning import Trainer

from model.transformer_decoder import MyTransformerDecoder
from data_loader.data_module import SimpleDataModule, WeldingDataModule
from post_hoc_ood import test_ood_detection
from utils import (
    get_logger,
    load_first_stage_model,
    get_laten_ds,
)
from vq_vae_transformer_ood import test_vq_vae_model, test_autoregressive_model
from mlflow_logger import MyMLFlowLogger


def init_transformer_model(conf: dict[str, any]):
    log.info("Initializing Transformer model with the following configuration:")
    first_stage_model_config = load_first_stage_model(conf["path_vq_vae"])
    # Retrieve parameters from conf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_embeddings = first_stage_model_config.num_embeddings
    n_cycles = conf["n_cycles"]

    enc_out_len = first_stage_model_config.enc_out_len

    # Compute dependent variables
    in_seq_len = (enc_out_len * n_cycles) + 1
    num_embeddings = num_embeddings + 3
    log.info(
        f"Input sequence length: {in_seq_len} | Number of classification classes: {conf['num_classes']} | Number of generation classes: {num_embeddings}"
    )

    # Initialize the Transformer model
    transformer = MyTransformerDecoder(
        dataset_name=conf["dataset"],
        d_model=conf["d_model"],
        embedding_classes=num_embeddings,
        seq_len=in_seq_len,
        n_blocks=conf["n_resblocks"],
        n_head=conf["n_heads"],
        n_classes=conf["num_classes"],
        res_dropout=conf["res_dropout"],
        att_dropout=conf["att_dropout"],
        learning_rate=conf["learning_rate"],
        class_h_bias=conf["use_class_head_bias"],
        n_cycles=conf["n_cycles"],
    ).to(device)

    return transformer


def get_new_trainer(
    epochs,
    logger,
    grad_clipping: float = 0.8,
    n_gpus=1,
    callbacks: list = [],
    use_cpu_only: bool = False,
):

    return Trainer(
        devices=n_gpus,
        accelerator="cpu" if use_cpu_only else "auto",
        num_nodes=1,
        max_epochs=epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=grad_clipping,
        strategy=DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else "auto",
        accumulate_grad_batches=5,
    )


def classification_finetuning(
    model,
    class_task_data_module,
    classification_epoch,
    logger,
    gradient_clip: float = 0.8,
    n_gpus: int = 1,
    use_cpu_only: bool = False,
):
    callbacks = get_callbacks(
        is_classification=True, use_early_stopping=True, save_model=True
    )
    model.switch_to_classification()

    trainer = Trainer(
        devices=n_gpus,
        num_nodes=1,
        accelerator="cpu" if use_cpu_only else "auto",
        logger=logger,
        callbacks=callbacks,
        max_epochs=classification_epoch,
        gradient_clip_val=gradient_clip,
        strategy=DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else "auto",
        accumulate_grad_batches=5,
    )
    trainer.fit(model, class_task_data_module)
    best_val_score = model.best_val_score
    print(f"Best val loss: {best_val_score}")
    best_path = None
    checkpoint_callback = None
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            checkpoint_callback = cb
            break

    if checkpoint_callback:
        best_path = checkpoint_callback.best_model_path
        print(f"Best model saved at: {best_path}")

    return best_path


def get_callbacks(
    is_classification: bool, use_early_stopping: bool, save_model: bool = False
):
    callbacks = []
    if is_classification:
        score = "val/f1_score"
        mode = "max"
    else:
        score = "val/loss"
        mode = "min"
    if save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath="model_checkpoints/VQ-VAE-transformer/",
            monitor=score,
            mode=mode,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
    if use_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=score, min_delta=0.001, patience=5, verbose=False, mode=mode
        )
        callbacks.append(early_stop_callback)
    return callbacks


def train_transformer(
    model, class_task_data_module, gen_task_data_module, conf: dict[str, any]
):
    use_cpu_only = conf["n_gpus"] == 0
    classification_epoch = conf["classification_epochs"]
    fine_tune_epochs = conf["finetune_epochs"]
    epoch_iter = conf["epoch_iter"]
    n_gpus = conf["n_gpus"]
    grad_clipping = conf["gradient_clip_val"]
    use_early_stopping = conf["use_early_stopping"]
    gen_epochs = conf["gen_epochs"]
    logger = get_logger(
        use_mlflow=conf["use_mlflow"], experiment_name=conf["mlflow_experiment_name"]
    )

    if conf["use_mlflow"]:
        logger.log_hyperparams(
            {
                "gradient_clip_val": conf["gradient_clip_val"],
                "model_name": conf["model_name"],
                "batch_size": conf["batch_size"],
                "epoch_iter": conf["epoch_iter"],
                "classification_epochs": conf["classification_epochs"],
                "finetune_epochs": conf["finetune_epochs"],
                "gen_epochs": conf["gen_epochs"],
                "seed": conf["seed"],
                "path_vq_vae": str(conf["path_vq_vae"]),
                "embedding_model_name": conf["embedding_model_name"],
                "prob_unk_token": conf["prob_unk_token"],
                "hyperparams_search_str": conf["hyperparams_search_str"],
            }
        )

    for epoch in range(epoch_iter):
        log.info("Genrerating stage")
        callbacks = get_callbacks(
            is_classification=False, use_early_stopping=use_early_stopping
        )
        trainer = get_new_trainer(
            epochs=gen_epochs,
            logger=logger,
            grad_clipping=grad_clipping,
            n_gpus=n_gpus,
            callbacks=callbacks,
            use_cpu_only=use_cpu_only,
        )
        model.switch_to_generate()
        trainer.fit(model, gen_task_data_module)

        if epoch == epoch_iter - 1:
            classification_finetuning(
                model=model,
                class_task_data_module=class_task_data_module,
                classification_epoch=fine_tune_epochs,
                gradient_clip=grad_clipping,
                logger=logger,
                n_gpus=n_gpus,
            )
        else:
            callbacks = get_callbacks(
                is_classification=True, use_early_stopping=use_early_stopping
            )
            trainer = get_new_trainer(
                epochs=classification_epoch,
                grad_clipping=grad_clipping,
                logger=logger,
                n_gpus=n_gpus,
                callbacks=callbacks,
                use_cpu_only=use_cpu_only,
            )
            log.info("Classification stage")
            model.switch_to_classification()
            trainer.fit(model, class_task_data_module)
    evaluate_ood_detection(model, class_task_data_module, conf, logger)
    

def evaluate_ood_detection(model: MyTransformerDecoder, class_task_data_module: SimpleDataModule, conf: dict[str, any], logger: MyMLFlowLogger):
    log.info("Testing OOD detection")
    model.switch_to_classification()
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=logger,
    )
    trainer.test(model, dataloaders=class_task_data_module.test_dataloader())
    if conf["compute_ood_score"]:
        device = "cuda:0"
        model.to(device)
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
            test_ood_detection(
                model=model,
                data_module=class_task_data_module,
                use_mlflow=conf["use_mlflow"]
            )
            test_autoregressive_model(
                model=model,
                data_module=class_task_data_module,
                device=device,
                use_mlflow=conf["use_mlflow"],
            )
            vq_vae_model = load_first_stage_model(conf["path_vq_vae"])
            vq_vae_model.to(device)
            test_vq_vae_model(
                data_path=Path(conf["dataset_path"]),
                vq_vae_model=vq_vae_model,
                classification_model=model,
                n_cycles=conf["n_cycles"],
                device=device,
                use_mlflow=conf["use_mlflow"],
            )
        finally:
            if conf["use_mlflow"]:
                log_context.__exit__(None, None, None)  # Manually exit the context


def get_data_modules(conf: dict[str, any]) -> tuple[SimpleDataModule, SimpleDataModule]:
    """Create data modules for generation and classification tasks.
    
    Args:
        conf: Configuration dictionary containing dataset parameters
        
    Returns:
        A tuple containing (class_task_data_module, gen_task_data_module)
    """
    (
        recon_train_ds,
        recon_val_ds,
        recon_test_ds,
        class_train_ds,
        class_val_ds,
        class_test_ds,
    ) = get_laten_ds(
        vq_vae_path=conf["path_vq_vae"],
        init_ds=conf["create_new_ds"],
        data_path=Path(conf["dataset_path"]),
        n_cycles=conf["n_cycles"],
        prob_unk_token=conf["prob_unk_token"],
        seq_prediction_task=True
    )

    gen_task_data_module = SimpleDataModule(
        train_ds=recon_train_ds,
        val_ds=recon_val_ds,
        test_ds=recon_test_ds,
        batch_size=conf["batch_size"],
    )

    class_task_data_module = SimpleDataModule(
        train_ds=class_train_ds,
        val_ds=class_val_ds,
        test_ds=class_test_ds,
        batch_size=conf["batch_size"],
    )
    return class_task_data_module, gen_task_data_module

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="Welding")
    args.add_argument("--model-name", type=str, default="Transformer")
    args.add_argument("--embedding-model-name", type=str, default="VQ-VAE")
    args.add_argument("--n-cycles", type=int, default=10)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--n-gpus", type=int, default=1)
    args.add_argument("--create-new-ds", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--d-model", type=int, default=512)
    args.add_argument("--batch-size", type=int, default=128)
    args.add_argument("--gradient-clip-val", type=float, default=0.8)
    args.add_argument("--prob-unk-token", type=float, default=0.0)
    args.add_argument("--learning-rate", type=float, default=1e-3)
    args.add_argument("--n-resblocks", type=int, default=6)
    args.add_argument("--n-heads", type=int, default=8)
    args.add_argument("--epoch-iter", type=int, default=3)
    args.add_argument("--gen-epochs", type=int, default=10)
    args.add_argument("--classification-epochs", type=int, default=2)
    args.add_argument("--finetune-epochs", type=int, default=5)
    args.add_argument("--att-dropout", type=float, default=0.0)
    args.add_argument("--res-dropout", type=float, default=0.1)
    args.add_argument("--mlflow-experiment-name", type=str, default="ood-welding")
    args.add_argument("--hyperparams-search-str", type=str, default="NoHyperparamSearch")
    args.add_argument("--dataset-path", type=str, default="data")
    args.add_argument("--use-mlflow", action=argparse.BooleanOptionalAction, default=True)
    args.add_argument("--use-class-head-bias", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--use-early-stopping", action=argparse.BooleanOptionalAction, default=True)
    args.add_argument("--compute-ood-score", action=argparse.BooleanOptionalAction, default=True)
    conf = vars(args.parse_args())
    log.info(conf)
    seed_everything(conf["seed"])

    conf["path_vq_vae"] = Path(
        "model_checkpoints/best_models/VQ-VAE/best_vq_vae.ckpt"
    )
    conf["num_classes"] = 2
    conf["data_dim"] = 2
    conf["model_name"] = conf["embedding_model_name"] + "_" + conf["model_name"]

    model = init_transformer_model(conf)

    class_task_data_module, gen_task_data_module = get_data_modules(conf)
    train_transformer(
        model=model,
        class_task_data_module=class_task_data_module,
        gen_task_data_module=gen_task_data_module,
        conf=conf,
    )


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

