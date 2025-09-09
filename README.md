# Out-of-Distribution Detection for Efficient Continual Learning in Quality Prediction for Arc Welding

**📄 Paper accepted at CIKM 2025** 

This repository contains the implementation for our research on out-of-distribution (OOD) detection in gas metal arc welding (GMAW) quality prediction. Our work addresses critical challenges in dynamic manufacturing environments where process parameters frequently change, causing distribution shifts that degrade model performance.

## Overview

Modern manufacturing environments are inherently dynamic, with frequent changes in setup, materials, and process parameters creating distribution shifts in sensor data. Traditional machine learning models trained on static distributions struggle to maintain performance across these shifts. Our approach leverages a VQ-VAE Transformer architecture to detect when new data significantly differs from the training distribution, triggering continual learning updates only when necessary.

### Key Contributions

- **Novel OOD Detection Method**: We extend the VQ-VAE Transformer architecture by using its autoregressive loss as a reliable OOD detection mechanism, outperforming conventional reconstruction methods and established baselines like MSP and ODIN.

- **Efficient Continual Learning**: By integrating OOD detection with continual learning strategies, we optimize model adaptation, triggering updates only when necessary and minimizing costly labeling requirements.

- **Real-world Validation**: Experimental validation in real-world welding scenarios demonstrates robust quality prediction capabilities across significant distribution shifts, particularly during transitions between different welding types (e.g., overlap to T-joints).

- **Quantitative Evaluation Metric**: Introduction of a novel metric that simultaneously evaluates OOD detection capability while interpreting in-distribution performance.

### Architecture

The system builds upon a state-of-the-art VQ-VAE Transformer model that combines:
- **Vector Quantized Variational Autoencoder (VQ-VAE)**: For learning compressed representations of welding sensor data
- **Transformer Architecture**: For capturing temporal dependencies in welding processes
- **Autoregressive Loss**: As an effective indicator for OOD detection
- **Continual Learning Integration**: Memory-aware adaptation strategies to prevent catastrophic forgetting

## Setup

```bash
uv sync
```

## Trainmodels

### MLP
```bash
python train_mlp.py
```

### Transformer
```bash
python train_transformer.py
```

### CODiT
```bash
python train_CODiT.py
```

## Continual Learning
```bash
python train_CODiT.py
```

## Dataset

The dataset can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15497262.svg)](https://doi.org/10.5281/zenodo.15497262).

### Setup Instructions

1. Download the dataset from the link above
2. Extract and place the dataset files in the `data/Welding` directory
3. Run the preprocessing script once to prepare the data:

```bash
python data/Welding/preprocess_data.py
```

**Note:** The preprocessing step is required before training any models.

## MLFlow

MLFlow is used for experiment tracking in this project. It helps track metrics, parameters, and artifacts across model training runs.

#### Setup

The project is configured to use a custom MLFlow server. Environment variables are automatically set up in the `mlflow_logger.py` file.

##### Environment Variables

You can configure MLFlow by setting the following environment variables:

```bash
# MLFlow server configuration
export MLFLOW_TRACKING_URL="http://mlflow.example.com:5000"
export MLFLOW_TRACKING_USERNAME="username"
export MLFLOW_TRACKING_PASSWORD="password"

# S3 artifact storage configuration
export MLFLOW_S3_ENDPOINT_URL="http://minio.example.com:9000"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio_password"
export AWS_BUCKET_NAME="mlflow"
```

If these variables are not set, the following defaults will be used:
- MLFLOW_TRACKING_URL: "http://localhost:5000"
- MLFLOW_S3_ENDPOINT_URL: "localhost:9000"
- AWS_BUCKET_NAME: "mlflow"

#### Running Experiments with MLFlow

To enable MLFlow logging when training models, add the `--use-mlflow` flag to your training command:

```bash
# MLP training with MLFlow
python train_mlp.py --use-mlflow

# Transformer training with MLFlow
python train_transformer.py --use-mlflow

# CODiT training with MLFlow
python train_CODiT.py --use-mlflow
```



## Results

| Model | F1 | Acc | AR | Rec | Enc | MSP | ODIN | Vac | Maha | ood_autoregressive_acc | ood_reconstruction_acc | ood_vq_encoder_acc | msp_ood_score_acc | odin_ood_score_acc | vacuity_ood_score_acc | mahalanobis_ood_score_acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CODIT | 0.44 ± 0.15 | *0.58* ± 0.07 | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| MLP | 0.52 ± 0.04 | 0.54 ± 0.03 | - | - | - | 0.28 ± 0.05 | *0.18* ± 0.03 | 0.20 ± 0.04 | *-0.00* ± 0.06 | - | - | - | **0.32** ± 0.04 | *0.20* ± 0.03 | 0.20 ± 0.04 | 0.05 ± 0.06 |
| MLP_EDL | *0.56* ± 0.11 | 0.57 ± 0.10 | - | - | - | *0.29* ± 0.07 | 0.18 ± 0.04 | *0.23* ± 0.07 | -0.01 ± 0.08 | - | - | - | *0.31* ± 0.08 | 0.19 ± 0.03 | *0.24* ± 0.05 | 0.04 ± 0.07 |
| VQ-VAE_MLP | 0.50 ± 0.02 | 0.51 ± 0.02 | - | - | - | 0.14 ± 0.03 | 0.17 ± 0.02 | 0.17 ± 0.02 | -0.03 ± 0.01 | - | - | - | 0.15 ± 0.02 | 0.17 ± 0.02 | 0.17 ± 0.02 | **0.07** ± 0.01 |
| VQ-VAE_Transformer | **0.65** ± 0.03 | **0.65** ± 0.03 | **0.35** ± 0.08 | **0.28** ± 0.07 | **0.20** ± 0.08 | **0.30** ± 0.03 | **0.25** ± 0.03 | **0.34** ± 0.02 | **0.05** ± 0.14 | **0.33** ± 0.11 | **0.26** ± 0.07 | **0.16** ± 0.06 | 0.30 ± 0.05 | **0.25** ± 0.03 | **0.32** ± 0.04 | *0.06* ± 0.07 |

*Note: Bold values indicate best performance, italic values indicate second-best.*

## Hyperparameter search

### Best Hyperparameters
| Model | batch_size | learning_rate | wgtDecay | momentum | wd | n_cycles | epochs | gradient_clip_val | n_hidden_layers | dropout_p | use_layer_norm | res_dropout | gen_epochs | finetune_epochs | epoch_iter | d_model |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CODIT | 128 | 0.005 | 1e-05 | 0.9 | 0.0005 | 10 | 40.0 | - | - | - | - | - | - | - | - | - |
| VQ-VAE_Transformer | 512 | 0.001 | - | - | - | 10 | - | 0.8 | - | - | - | 0.1 | 10.0 | 5.0 | 2.0 | 128 |
| MLP | 512 | 0.005 | - | - | - | 10 | 50.0 | 1.5 | 4.0 | 0.2 | True | - | - | - | - | 512 |
| MLP_EDL | 512 | 0.005 | - | - | - | 10 | 50.0 | 0.5 | 4.0 | 0.1 | False | - | - | - | - | 128 |
| VQ-VAE_MLP | 512 | 0.0005 | - | - | - | 5 | 50.0 | 0.5 | 3.0 | 0.0 | False | - | - | - | - | 512 |

### Hyperparameter Search Ranges

| Model | Hyperparameter | Search Range |
|-------|---------------|--------------|
| VQ-VAE | gradient-clip-val | [0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0] |
| | learning-rate | [0.001, 0.0005, 0.0001] |
| | num-embeddings | [128, 256, 512] |
| | embedding-dim | [2, 32, 64, 128, 256, 512] |
| | hidden-dim | [128, 256, 512] |
| | batch-size | [128, 256, 512] |
| | n-resblocks | [4, 6, 8] |
| | dropout-p | [0.05, 0.1, 0.2, 0.3] |
| | epochs | [50] |
| VQ-VAE_MLP | n-cycles | [1, 5, 10] |
| | epochs | [50] |
| | batch-size | [256, 512, 1024, 2048] |
| | gradient-clip-val | [0.5, 0.7, 0.8, 0.9, 1.0, 1.5] |
| | learning-rate | [0.0005, 0.001, 0.005] |
| | hidden-dim | [32, 64, 128, 256, 512, 1024] |
| | n-hidden-layers | [2, 3, 4, 5] |
| | dropout-p | [0.0, 0.1, 0.15, 0.2, 0.3] |
| | use-layer-norm | [True, False] |
| Transformer | d-model | [128, 256, 512] |
| | batch-size | [256, 512] |
| | gradient-clip-val | [0.5, 0.7, 0.8, 0.9, 1.0, 1.5] |
| | learning-rate | [0.0005, 0.001, 0.005] |
| | res-dropout | [0.05, 0.1, 0.2, 0.3] |
| | n-resblocks | [4, 6, 8] |
| | n-cycles | [1, 5, 10] |
| | n-heads | [4, 8] |
| | classification-epoch | [2, 4] |
| | gen-epochs | [10] |
| | finetune-epochs | [5, 10] |
| | epoch-iter | [2, 3, 4] |
| MLP | n-cycles | [1, 5, 10] |
| | epochs | [50] |
| | batch-size | [256, 512, 1024, 2048] |
| | gradient-clip-val | [0.5, 0.7, 0.8, 0.9, 1.0, 1.5] |
| | learning-rate | [0.0005, 0.001, 0.005] |
| | hidden-dim | [32, 64, 128, 256, 512, 1024] |
| | n-hidden-layers | [2, 3, 4, 5] |
| | dropout-p | [0.0, 0.1, 0.15, 0.2, 0.3] |
| | use-layer-norm | [True, False] |
| MLP_EDL | n-cycles | [1, 5, 10] |
| | epochs | [50] |
| | batch-size | [256, 512, 1024, 2048] |
| | gradient-clip-val | [0.5, 0.7, 0.8, 0.9, 1.0, 1.5] |
| | learning-rate | [0.0005, 0.001, 0.005] |
| | hidden-dim | [32, 64, 128, 256, 512, 1024] |
| | n-hidden-layers | [2, 3, 4, 5] |
| | dropout-p | [0.0, 0.1, 0.15, 0.2, 0.3] |
| | use-layer-norm | [True, False] |
| | annealing-start | [0.001, 0.002, 0.003, 0.004, 0.005] |
| CODiT | lr | [0.005, 0.001, 0.0005, 0.0001] |
| | wgtDecay | [0.0001, 0.00005, 0.00001] |
| | momentum | [0.9] |
| | wd | [0.0005] |
| | workers | [4] |
| | n-cycles | [1, 5, 10] |
| | batch-size | [128, 256, 512, 1024] |
| | epochs | [25, 40, 50] |
| | seed | [42] |

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@inproceedings{hahn2025ood,
  title={Out of Distribution Detection for Efficient Continual Learning in Quality Prediction for Arc Welding},
  author={Hahn, Yannik and Voets, Jan and K{\"o}nigsfeld, Antonin and Tercan, Hasan and Meisen, Tobias},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  year={2025},
  organization={ACM},
  note={Accepted at CIKM 2025}
}
```

**Paper**: [https://arxiv.org/html/2508.16832v1](https://arxiv.org/html/2508.16832v1)

## Abstract

Modern manufacturing relies heavily on fusion welding processes, including gas metal arc welding (GMAW). Despite significant advances in machine learning-based quality prediction, current models exhibit critical limitations when confronted with the inherent distribution shifts that occur in dynamic manufacturing environments. In this work, we extend the VQ-VAE Transformer architecture—previously demonstrating state-of-the-art performance in weld quality prediction—by leveraging its autoregressive loss as a reliable out-of-distribution (OOD) detection mechanism. Our approach exhibits superior performance compared to conventional reconstruction methods, embedding error-based techniques, and other established baselines. By integrating OOD detection with continual learning strategies, we optimize model adaptation, triggering updates only when necessary and thereby minimizing costly labeling requirements. We introduce a novel quantitative metric that simultaneously evaluates OOD detection capability while interpreting in-distribution performance. Experimental validation in real-world welding scenarios demonstrates that our framework effectively maintains robust quality prediction capabilities across significant distribution shifts, addressing critical challenges in dynamic manufacturing environments where process parameters frequently change.

## Authors

- **Yannik Hahn** - Institute for Technologies and Management of Digital Transformation (TMDT), University of Wuppertal
- **Jan Voets** - Institute for Technologies and Management of Digital Transformation (TMDT), University of Wuppertal  
- **Antonin Königsfeld** - Institute for Technologies and Management of Digital Transformation (TMDT), University of Wuppertal
- **Hasan Tercan** - Institute for Technologies and Management of Digital Transformation (TMDT), University of Wuppertal
- **Tobias Meisen** - Institute for Technologies and Management of Digital Transformation (TMDT), University of Wuppertal

*Corresponding author: yhahn@uni-wuppertal.de*