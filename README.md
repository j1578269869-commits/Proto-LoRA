# Proto-LoRA: Prototype-based LoRA Fine-tuning for Multimodal Information Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Proto-LoRA is an implementation framework for efficient multimodal relation extraction and entity typing using Low-Rank Adaptation (LoRA) in large multimodal models. It emphasizes prototype-based semantic guidance for entity and relation embeddings, while preserving the lightweight tuning characteristics of LoRA.

This repository is intended as a reproducible codebase, with a focus on configuration-driven execution, model modularity, and clear experiment workflow.

## Introduction

Modern multimodal relation extraction systems must integrate visual and textual context while remaining practical for large pre-trained multimodal models. Proto-LoRA addresses this challenge by:

- using prototype embeddings to represent entity and relation semantics,
- applying LoRA to reduce trainable parameter counts,
- supporting multiple backbone models including BLIP2, LLaVA, and Qwen2-VL.

The implementation supports two primary tasks:

1. **Multimodal Entity Typing (MET)**
2. **Multimodal Relation Extraction (MRE)**

## Design Principles

- **Reproducibility**: all experiments are driven by configuration files in `config/`.
- **Modularity**: model-specific training logic is separated into dedicated scripts.
- **Transparency**: no experimental numbers are presented unless derived from validated runs.
- **Open-source readiness**: environment setup and data dependencies are documented.

## Installation

### Requirements

- Python 3.8 or later
- CUDA 11.x / 12.x compatible GPU
- `pip`

### Setup

```bash
git clone https://github.com/j1578269869-commits/Proto-LoRA.git
cd Proto-LoRA
pip install -r requirements.txt
```

### Hugging Face Authentication

If model weights or datasets are hosted on Hugging Face, export your token:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

## Repository Layout

- `config/` — experiment configuration templates
- `data/` — data preparation and format instructions
- `scripts/` — helper scripts for launching experiments and downloads
- `embedding_entity/` — entity prototype embedding utilities
- `embedding_relation/` — relation prototype embedding utilities
- `LoRA/` — LoRA adaptation components and helper modules
- `Zero-shot/` — zero-shot evaluation utilities
- `trainlora_*.py` — training scripts for each model/task
- `run_MET.sh`, `run_MRE.sh` — execution wrappers
- `requirements.txt` — dependency list
- `README.md` — repository documentation

## Configuration

Proto-LoRA uses YAML configuration files. A configuration file should specify at least:

- `model_path` or `model_name`
- `prototype_path`
- `train_path`, `val_path`, `test_path`
- `image_base_path`
- `save_dir`
- training hyperparameters (`epochs`, `batch_size`, `lr`, etc.)

Example configuration for MRE:

```yaml
model_path: "/path/to/blip2-flan-t5-xxl"
prototype_path: "/path/to/extract_relation_embedding_blip.pt"
train_path: "/path/to/mnre_train.json"
val_path: "/path/to/mnre_val.json"
test_path: "/path/to/mnre_test.json"
image_base_path: "/path/to/images"
save_dir: "./outputs/mre_blip2"
epochs: 3
batch_size: 1
lr: 1e-4
```

## Running Experiments

### Launch from shell scripts

The repository provides wrapper scripts for convenience:

```bash
bash run_MRE.sh
bash run_MET.sh
```

These scripts should be configured to point to the desired YAML file before execution.

### Run a specific model/task

```bash
python trainlora_blip2_MRE.py --config config/mre_blip2.yaml
python trainlora_llava_MET.py --config config/met_llava.yaml
python trainlora_Qwen2_MRE.py --config config/mre_qwen.yaml
```

### Command-line overrides

Most scripts support overriding configuration fields through CLI arguments. Use this to run quick ablations without editing YAML files.

## Evaluation and Output

During training, the code produces:

- saved adapter weights in `save_dir`
- JSON-based logs for training history
- validation and test predictions
- evaluation metrics computed on the current dataset

The exact output filenames and structure are determined by the training script and the configuration.

## Notes on Experimental Results

This README does not include unverified performance figures. For publication-quality documentation, populate this section with metrics obtained from reproducible runs and documented evaluation protocols.

When adding results, include:

- task description and dataset split details
- evaluation metrics (precision, recall, F1, accuracy)
- comparison to baseline methods
- any ablation or probe study settings

## Data Preparation

Ensure your dataset files match the format expected by the scripts.

- `train_path`, `val_path`, `test_path` should be valid JSON files.
- `image_base_path` should point to the root folder containing image assets.
- prototype embedding files should be generated through the provided embedding utilities.

If you use third-party datasets, cite the source and describe any preprocessing steps.

## Best Practices for Publication

1. Keep all dataset and model paths outside the codebase.
2. Use configuration files for hyperparameters and paths.
3. Document the hardware used for experiments.
4. Report average results over multiple random seeds if possible.
5. Include the exact command used for each key result.

## Contribution Guidelines

Contributions are welcome. A high-quality contribution should include:

- a clear description of the change
- reproducible commands or config examples
- code that avoids hard-coded paths
- documentation updates when behavior changes

## Citation

If Proto-LoRA is used in your research, please cite the repository and describe the method in your own paper.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
