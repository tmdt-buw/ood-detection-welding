import argparse
import optuna
from optuna.samplers import RandomSampler
import yaml
import subprocess
import os
import logging as log
from datetime import datetime
import signal
import sys
from multiprocessing import Pool


def shutdown_handler(sig, frame):
    """Handle shutdown signals by gracefully exiting the hyperparameter search process.
    
    Args:
        sig: The signal number
        frame: The current stack frame
    """
    print(f"Caught signal {sig}, stopping hyperparameter search...")
    sys.exit(0)


def load_params_from_yaml(yaml_file: str):
    """Load hyperparameters from a YAML configuration file.
    
    Args:
        yaml_file (str): Path to the YAML configuration file
        
    Returns:
        dict: Loaded hyperparameters
    """
    with open(yaml_file, "r") as file:
        params = yaml.safe_load(file)
    return params


def build_trial_from_yaml(yaml_file: str, trial: optuna.Trial):
    """Build Optuna trial parameters from YAML configuration.
    
    Args:
        yaml_file (str): Path to YAML configuration file
        trial (optuna.Trial): Current Optuna trial
        
    Returns:
        dict: Trial parameters
    """
    yaml_dict = load_params_from_yaml(yaml_file)

    params = {}
    for var_name, var_value_dict in yaml_dict.items():
        var_value_dict_keys = list(var_value_dict.keys())
        if var_value_dict_keys == ["values"] and isinstance(
            var_value_dict["values"], list
        ):
            params[var_name] = trial.suggest_categorical(
                var_name, var_value_dict["values"]
            )
        elif "min" in var_value_dict_keys and "max" in var_value_dict_keys:
            min = var_value_dict["min"]
            max = var_value_dict["max"]
            if "distribution" in var_value_dict_keys:
                distribution = var_value_dict["distribution"]
            else:
                distribution = "uniform"

            if distribution == "uniform":
                params[var_name] = trial.suggest_float(var_name, min, max)
            elif distribution == "loguniform":
                params[var_name] = trial.suggest_float(var_name, min, max, log=True)
            elif distribution == "discrete_uniform":
                params[var_name] = trial.suggest_discrete_uniform(
                    var_name, min, max, q=1
                )
            elif distribution == "int":
                params[var_name] = trial.suggest_int(var_name, min, max)
            elif distribution == "int_uniform":
                params[var_name] = trial.suggest_int(var_name, min, max)
            elif distribution == "categorical":
                params[var_name] = trial.suggest_categorical(
                    var_name, list(range(min, max + 1))
                )
            else:
                raise ValueError(f"Distribution not supported: {distribution}")
        else:
            raise ValueError(
                f"Parameter not supported {var_name}: {var_value_dict.keys()}"
            )

    return params


def objective(
    trial: optuna.Trial,
    config_file_path: str,
    python_script: str,
    hyperparams_search_str: str,
    create_new_ds: bool = False,
):
    """Run a single trial of hyperparameter optimization.
    
    Args:
        trial (optuna.Trial): Current Optuna trial
        config_file_path (str): Path to YAML config file
        python_script (str): Path to training script
        hyperparams_search_str (str): Search string for hyperparameters
        create_new_ds (bool): Whether to create a new dataset
        
    Returns:
        float: Best validation metric (F1 score for classification tasks)
    """
    params = build_trial_from_yaml(config_file_path, trial)

    # Prepare the command
    command = [
        "python",
        python_script,
        "--seed",
        str(trial.number + 42),  # Use trial number as seed offset
    ]
    
    if create_new_ds:
        command.append("--create-new-ds")
    
    # Add parameters to command
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                command += [f"--{key}"]
            else:
                command += [f"--no-{key}"]
        else:
            command += [f"--{key}", str(value)]

    trial_id = trial.number
    study_name = trial.study.study_name
    log.info("-" * 20)
    log.info(f"Study: {study_name} Trial {trial_id}")
    log.info(f"Time: {trial.datetime_start}")
    log.info(f"Command: {' '.join(command)}")

    # Run the script
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the standard output and errors
    log.info(result.stdout)
    if result.stderr:
        log.error(result.stderr)
    
    # Extract the best validation metric from the subprocess output
    best_val_metric = None
    for line in result.stdout.splitlines():
        if "Best val loss:" in line:
            # For classification tasks in this repo, higher F1 score is better
            best_val_metric = float(line.split("Best val loss:")[-1].strip())
            break
    
    if best_val_metric is None and result.returncode == 0:
        raise ValueError("Best validation metric not found in the output")

    # Only save if there was an error or if the trial was successful
    if result.returncode != 0 or best_val_metric is not None:
        os.makedirs(f"logs/optuna/{study_name}", exist_ok=True)
        stdout_filename = f"logs/optuna/{study_name}/trial_{trial_id}_stdout.txt"
        stderr_filename = f"logs/optuna/{study_name}/trial_{trial_id}_stderr.txt"
        # Write stdout and stderr to files
        with open(stdout_filename, "w") as f:
            f.write(result.stdout)
        with open(stderr_filename, "w") as f:
            f.write(result.stderr)
    
    log.info(f"Best val metric: {best_val_metric}")
    return best_val_metric


def run_study(study_params: dict[str, any]) -> dict[str, any]:
    """
    Run a single study with given parameters and return the best parameters and score

    Steps:
    1. Create a study with the given parameters
    2. Run the study and optimize the objective function
    3. Return the best parameters and score

    Args:
        study_params (dict): Parameters for the study

    Returns:
        dict: Best parameters and score
    """
    params = argparse.Namespace(**study_params)
    hyperparams_search_str = f"hyperparams_search_{study_params['study_name']}"
    seed = study_params["seed"]
    
    # Determine optimization direction based on script
    # For train_transformer.py, we want to minimize loss or maximize f1_score
    direction = "maximize" if "train_vq_vae" not in params.python_script else "minimize"
    
    # Create study with storage
    storage_url = "sqlite:///optuna_studies.db"
    study = optuna.create_study(
        study_name=study_params["study_name"], 
        storage=storage_url,
        load_if_exists=True,  # This is key - allows multiple processes to use same study
        sampler=RandomSampler(seed=seed),  # Give each process a different seed
        direction=direction,
    )
    
    study.optimize(
        func=lambda trial: objective(
            trial=trial,
            config_file_path=params.config_file_path,
            python_script=params.python_script,
            hyperparams_search_str=hyperparams_search_str,
            create_new_ds=params.create_new_ds,
        ),
        n_trials=params.n_trials // params.n_parallel,
    )

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "hyperparams_search_str": hyperparams_search_str,
    }


def main(params):
    """Run parallel hyperparameter optimization studies.
    
    Steps:
    1. Create multiple parameter sets with different random seeds
    2. Run parallel optimization studies using multiprocessing
    3. Find the best configuration across all studies
    4. Train the best model 5 times with different seeds
    5. Save results and logs for each training run
    
    Args:
        params: Command line arguments containing study configuration
        
    Returns:
        None
    """
    # Create multiple parameter sets with different configurations
    study_params_list = []
    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    base_params = vars(params)
    for i in range(params.n_parallel):
        study_params = base_params.copy()
        study_params["seed"] = i
        study_params["study_name"] = f"study_{datetime_str}_{i}"
        study_params_list.append(study_params)

    if params.n_parallel == 1:
        results = [run_study(study_params_list[0])]
    else:
        # Run studies in parallel and collect results
        with Pool(processes=len(study_params_list)) as pool:
            results = pool.map(run_study, study_params_list)

    # Determine optimization direction for finding best result
    is_maximize = "train_vq_vae" not in params.python_script
    
    # Find the best configuration across all parallel studies
    if is_maximize:
        best_result = max(results, key=lambda x: x["best_value"])
    else:
        best_result = min(results, key=lambda x: x["best_value"])
        
    best_params = best_result["best_params"]
    hyperparams_search_str = best_result["hyperparams_search_str"]

    log.info(f"Best overall parameters: {best_params}")
    log.info(f"Best overall value: {best_result['best_value']}")


    if "train_vq_vae" in params.python_script:
        return
        
    # remove seed from best_params
    if "seed" in best_params:
        best_params.pop("seed")
    
    # Train the best model 5 times
    for seed in range(5):
        log.info(f"Training best model - Run {seed+1}/5 with seed {seed}")
        command = [
            "python",
            params.python_script,
            "--seed",
            str(seed),
            "--use-mlflow",
            "--compute-ood-score",
        ]
        
        if params.create_new_ds:
            command.append("--create-new-ds")

        # Add best parameters to command
        for key, value in best_params.items():
            if isinstance(value, bool):
                if value:
                    command += [f"--{key}"]
                else:
                    command += [f"--no-{key}"]
            else:
                command += [f"--{key}", str(value)]

        log.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        # Log outputs
        log.info(result.stdout)
        if result.stderr:
            log.error(result.stderr)

        # Save results
        os.makedirs(f"logs/best_model/{hyperparams_search_str}", exist_ok=True)
        stdout_filename = (
            f"logs/best_model/{hyperparams_search_str}/run_{i+1}_stdout.txt"
        )
        stderr_filename = (
            f"logs/best_model/{hyperparams_search_str}/run_{i+1}_stderr.txt"
        )
        with open(stdout_filename, "w") as f:
            f.write(result.stdout)
        with open(stderr_filename, "w") as f:
            f.write(result.stderr)


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Hyperparameter Search")
    parser.add_argument("--n-trials", type=int, default=2, help="Total number of trials to run")
    parser.add_argument("--python-script", type=str, default="train_mlp.py", help="Script to run for training")
    parser.add_argument("--config-file-path", type=str, default="sweep_files/mlp.yml", help="Path to YAML config file")
    parser.add_argument("--create-new-ds", action=argparse.BooleanOptionalAction, default=False, help="Whether to create a new dataset")
    parser.add_argument("--n-parallel", type=int, default=1, help="Number of parallel studies to run")
    args = parser.parse_args()

    assert (
        args.n_trials % args.n_parallel == 0 and args.n_trials >= args.n_parallel
    ), f"Number of trials must be divisible by number of parallel studies and greater than number of parallel studies"

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    main(args)