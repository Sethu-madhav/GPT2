# Pre-training a GPT Style Language Model from Scratch 

This repository contains a comprehensive framework for pre-training a GPT style language model from the ground up in PyTorch. The project originated as an implementation of Andrej Karpathy's NanoGPT project and has since been significantly extended to support large-scale dataset processing, formal model evaluation, and modern development tools.

The framework is capable of processing the **FineWeb** dataset, training a model in a distributed fashion using FSDP, and evaluating its performance on the **HellaSwag** commonsense reasoning benchmark.

***

## Key Features

* **Large-Scale Data Processing**: Includes scripts to download and stream the massive FineWeb dataset efficiently.
* **Pre-Tokenization**: A dedicated script to tokenize the dataset ahead of time, accelerating the training pipeline.
* **Integrated Evaluation**: The training loop incorporates validation on the HellaSwag benchmark to track model performance on a downstream task.
* **Distributed Training**: Built with support for PyTorch FSDP to train on multi-GPU systems (e.g., 2x NVIDIA 5090s).
* **Modern Tooling**: Uses `uv` and `pyproject.toml` for fast and reliable dependency management.

***

## Getting Started

Follow these steps to set up the environment, prepare the data, and run the training.

### 1. Installation

This project uses `uv` for environment and dependency management.

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
cd [your-repo-name]

# Create a virtual environment and install dependencies with uv
uv venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
uv pip install -r requirements.txt
```
### 2. Data Preparation (FineWeb)
Training is performed on a pre-tokenized version of the FineWeb dataset.

Step A: Download the raw data
The `fineweb.py` script downloads and shards the dataset into smaller files.

```bash
python fineweb.py
```

### 3. Training the Model
The `train_gpt2.py` script is the main entry point for training. To launch a distributed training job using FSDP on 2 GPUs, use `torchrun`:
```bash
# Launch training on  2x NVIDIA 5090s GPUs
torchrun --nproc_per_node=2 train_gpt2.py
```
The script will periodically report training loss and HellaSwag validation accuracy. Model checkpoints will be saved in the `log/` directory.

### 4. Generating Text (Inference)
To interact with your trained model, use the `play.ipynb` Jupyter notebook. It provides a simple interface to load a checkpoint and generate text samples.






