# Low-Rank Adaptation for Fine-Tuning LLMs

This project implements Low-Rank Adaptation (LoRA) for large language models. The project is structured for
modularity and ease of use, allowing for easy downloading, preprocessing, and training of models with various datasets.

## Structure

```
LoRA_Project/
├── data/
│   ├── __init__.py
│   ├── downloader.py
│   ├── preprocessor.py
│   └── dataloader.py
│
├── models/
│   ├── __init__.py
│   ├── loader.py
│   ├── lora.py
│   └── LLMs/
│       ├── gpt2.py
│       ├── llama.py
│       └── other.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── evaluation.py
│
├── utils/
│   ├── __init__.py
│   ├── config.py
│   └── logger.py
│
├── main.py
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/qlinhta/llm-psl.git
   cd lora
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The script can be run with various command-line arguments to specify the dataset, model, tokenizer, and other
hyperparameters.

```bash
python main.py --dataset 1 --model gpt2 --tokenizer gpt2 --epochs 3 --batch_size 16 --learning_rate 0.0001 --rank 4
```

Sure, here is a more professional and helper-like rewrite for that section:

## Usage

To run the script, use the following command-line arguments to specify the dataset, model, tokenizer, and other
hyperparameters. Below is an example command followed by a detailed description of each argument.

### Example Command

```bash
python main.py --dataset 1 --model gpt2 --tokenizer gpt2 --epochs 3 --batch_size 16 --learning_rate 0.0001 --rank 4
```

### Arguments

- `--dataset <DATASET_ID>`: Specifies the ID of the dataset to use. For example:
    - `1` for SST-2 (Stanford Sentiment Treebank)
    - `2` for MRPC (Microsoft Research Paraphrase Corpus)
    - `3` for RTE (Recognizing Textual Entailment)
    - `4` for CoLA (Corpus of Linguistic Acceptability)
    - `5` for STS-B (Semantic Textual Similarity Benchmark)

- `--model <MODEL_NAME>`: The name of the pre-trained model to be used. Common examples include:
    - `gpt2` for GPT-2
    - `llama` for LLaMA

- `--tokenizer <TOKENIZER_NAME>`: The name of the tokenizer corresponding to the model. This should match the model
  being used (e.g., `gpt2` for GPT-2).

- `--epochs <NUM_EPOCHS>`: Specifies the number of training epochs. Example: `3`.

- `--batch_size <BATCH_SIZE>`: Defines the batch size for training. Example: `16`.

- `--learning_rate <LEARNING_RATE>`: Sets the learning rate for the optimizer. Example: `0.0001`.

- `--rank <LORA_RANK>`: Specifies the rank for the Low-Rank Adaptation (LoRA). Example: `4`.

**Example**:

Running with SST-2 Dataset and GPT-2 Model

```bash
python3 main.py --dataset 1 --model gpt2 --tokenizer gpt2 --epochs 3 --batch_size 16 --learning_rate 0.0001 --rank 4
```

Running with MRPC Dataset and LLaMA Model

```bash
python3 main.py --dataset 2 --model llama --tokenizer llama --epochs 5 --batch_size 32 --learning_rate 0.00005 --rank 4
```