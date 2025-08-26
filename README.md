# Building a GPT-style Language Model from Scratch

This repository contains the code and notebooks for building and training a GPT-style language model from the ground up, following the educational series by Andrej Karpathy. The project explores the fundamental components of the transformer architecture, starting with a simple Bigram model and progressing to a full GPT implementation.

## About This Project

The primary goal of this project is to gain a deep, hands-on understanding of the architecture that powers modern Large Language Models (LLMs). By implementing each component step-by-step in PyTorch, from the simplest baseline to a complete transformer model, this project serves as a practical learning exercise.

The code is heavily inspired by and based on Andrej Karpathy's "Let's build GPT: from scratch, in code, spelled out." video tutorial.

## Getting Started

To run the notebooks and scripts in this repository, follow these steps.

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Sethu-madhav/GPT2.git](https://github.com/Sethu-madhav/GPT2.git)
    cd GPT2
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    The `requirements.txt` file contains all the necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure and Content

This repository is organized into two main Jupyter notebooks:

### 1. `BigramLanguageModel.ipynb`

This notebook serves as the starting point and implements a simple Bigram language model. It's a character-level model that predicts the next character based only on the immediately preceding character. This helps to establish a baseline and understand the fundamental concepts of language modeling and tokenization.

### 2. `GPT_Language_Model.ipynb`

This is the core of the project. This notebook builds a full decoder-only transformer model, similar in architecture to GPT-2. It implements all the key components from scratch, including:

-   **Token and Positional Embeddings**
-   **Self-Attention and Multi-Head Attention**
-   **Causal Masking** (to prevent the model from looking ahead)
-   **Transformer Blocks** (with residual connections and layer normalization)
-   **Feed-Forward Networks**

The notebook walks through the training process and demonstrates how to generate new text from the trained model.

## What I Learned

-   A practical understanding of the self-attention mechanism, the core of the transformer architecture.
-   The importance of positional encodings for sequence data.
-   The role of residual connections and layer normalization in training deep neural networks.
-   The process of autoregressive text generation.
-   How a complex model like GPT can be broken down into understandable and implementable parts.

## Acknowledgments

A huge thank you to **Andrej Karpathy** for his excellent "Neural Networks: Zero to Hero" series, which made this project possible.

-   **[Link to the YouTube video: "Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY)**

