from pathlib import Path
import logging
import argparse
import torch

from utils import get_logger
from mlflow_logger import MyMLFlowLogger	
from vq_vae_transformer_ood import MyTransformerDecoder
from model.mlp import MLP
from train_transformer import evaluate_ood_detection as evaluate_ood_detection_transformer, get_data_modules as get_data_modules_transformer
from train_mlp import test_model as evaluate_ood_detection_mlp, get_data_module as get_data_module_mlp
from train_CODiT import evaluate_model as evaluate_CODiT


def evaluate_transformer_model(model_name: str, seed: int, checkpoint: Path, mlflow_logger: MyMLFlowLogger, device: str, use_mlflow: bool):
    """Evaluate a transformer-based model on out-of-distribution data.
    
    Args:
        model_name: Name of the model to evaluate
        seed: Random seed used for training
        checkpoint: Path to the model checkpoint	
        mlflow_logger: Logger for tracking metrics
        device: Device to run the model on
        use_mlflow: Whether to use mlflow
    """

    model = MyTransformerDecoder.load_from_checkpoint(checkpoint, map_location=device)

    conf = {
        "model_name" : model_name,
        "dataset_path": Path("data"),
        "batch_size": 128,
        "n_cycles": model.n_cycles,
        "prob_unk_token": 0.0,
        "create_new_ds": False,
        "use_mlflow": use_mlflow,
        "seed": seed,
        "compute_ood_score": True,
        "path_vq_vae": Path("model_checkpoints/best_models/VQ-VAE/best_vq_vae.ckpt"),
    }
    data_module, _ = get_data_modules_transformer(conf)
    logging.info(f"Conf: {conf}")
    logging.info(f"Model: {model.device}")
    evaluate_ood_detection_transformer(model, data_module, conf, mlflow_logger)




def evaluate_mlp_model(model_name: str, seed: int, checkpoint: Path, mlflow_logger: MyMLFlowLogger, device: str, use_mlflow: bool):
    """Evaluate an MLP-based model on out-of-distribution data.
    
    Args:
        model_name: Name of the model to evaluate
        seed: Random seed used for training
        checkpoint: Path to the model checkpoint
        mlflow_logger: Logger for tracking metrics
        device: Device to run the model on
        use_mlflow: Whether to use mlflow
    """
    model = MLP.load_from_checkpoint(checkpoint, map_location=device)

    conf = {
        "model_name" : model_name,
        "data_path": "data",
        "batch_size": 512,
        "prob_unk_token": 0.0,
        "create_new_ds": False,
        "use_mlflow": use_mlflow,
        "compute_ood_score": True,
        "discret_model": "",
        "seed": seed,
        "use_edl_loss": model.use_edl_loss,
        "path_vq_vae": Path("model_checkpoints/best_models/VQ-VAE/best_vq_vae.ckpt"),
    }
    if model.use_latent_input:
        conf["n_cycles"] = (model.input_size - 1) // 16
        conf["discret_model"] = "VQ-VAE"
    else:
        conf["n_cycles"] = model.input_size // 200


    data_module = get_data_module_mlp(conf, Path("data/Welding"))
    logging.info(f"Conf: {conf}")

    evaluate_ood_detection_mlp(
        model=model, 
        data_module=data_module, 
        conf=conf, 
        logger=mlflow_logger,
    )

def evaluate_codit_model(model_name: str, seed: int, checkpoint: Path, mlflow_logger: MyMLFlowLogger, device: str, use_mlflow: bool):
    """Evaluate a CODiT-based model on out-of-distribution data.
    
    python train_CODiT.py --seed 4 --use-mlflow --compute-ood-score --lr 0.005 --wgtDecay 0.0001 --momentum 0.9 --wd 0.0005 --workers 4 --transformation_list dilation --n-cycles 5 --batch-size 256 --epochs 25 --std-factor 0.25
    Args:
        model_name: Name of the model to evaluate
        seed: Random seed used for training
        checkpoint: Path to the model checkpoint
        mlflow_logger: Logger for tracking metrics
        device: Device to run the model on
        use_mlflow: Whether to use mlflow
    """
    conf = {
        "discret_model": "",
        "model_name" : model_name,
        "data_path": "data",
        "use_mlflow": use_mlflow,
        "compute_ood_score": True,
        "seed": seed,
        "transformation_list": ["dilation", "erosion", "identity"],
        "transformation_list_len": 3,
        "wl": 200*10,
        "wd": 0.0005,
        "n_cycles": 10,
        "batch_size": 512,
        "std_factor": 1,
        "num_classes": 2,
    }

    args = argparse.Namespace(**conf)
    data_module = get_data_module_mlp(conf, Path("data/Welding"))

    evaluate_CODiT(
            args=args,
            best_model_path=Path(checkpoint),
            device=device,
            data_module=data_module,
            logger=mlflow_logger,
            std_factor=args.std_factor
        )


def main():
    """Run evaluation of different models on out-of-distribution data.
    
    Loads model checkpoints for each model type and seed, then evaluates
    their performance on OOD data using the appropriate evaluation function.
    """
    use_mlflow = True
    model_list = [
        # "VQ-VAE_Transformer",
        "CODiT",
    ]

    checkpoint_path = Path("model_checkpoints/best_models")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name in model_list:
        for seed in range(0, 5):
            model_path = checkpoint_path / model_name / f"seed_{seed}"

            checkpoints = list(model_path.glob("**/*.ckpt"))
            logging.info(f"Found {len(checkpoints)} checkpoints for {model_name} with seed {seed}")

            checkpoint = [f for f in checkpoints if "last" in f.name]

            if len(checkpoint) == 0:
                checkpoints = list(model_path.glob("**/*.pt"))
                checkpoint = [f for f in checkpoints if "last" in f.name]
                if len(checkpoint) == 0:
                    logging.warning(f"No checkpoint found for {model_name} with seed {seed}")
                    continue
            checkpoint = checkpoint[0]
            logging.info(f"Evaluating {model_name} with seed {seed} from checkpoint {checkpoint}")

            if model_name == "VQ-VAE_Transformer":
                mlflow_logger = get_logger(use_mlflow=use_mlflow, experiment_name="ood-welding-test")
                evaluate_transformer_model(model_name, seed, checkpoint, mlflow_logger, device, use_mlflow)
            elif model_name == "CODiT":
                mlflow_logger = MyMLFlowLogger(experiment_name="ood-welding-test", run_name=f"{model_name}-seed_{seed}")
                mlflow_logger.start_run()
                evaluate_codit_model(model_name, seed, checkpoint, mlflow_logger, device, use_mlflow)
            else:
                mlflow_logger = get_logger(use_mlflow=use_mlflow, experiment_name="ood-welding-test")
                evaluate_mlp_model(model_name, seed, checkpoint, mlflow_logger, device, use_mlflow)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.set_float32_matmul_precision('medium')
    main()
