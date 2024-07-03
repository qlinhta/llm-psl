# Low-Rank Adaptation for Fine-Tuning LLMs

This project implements Low-Rank Adaptation (LoRA) for large language models. The project is structured for
modularity and ease of use, allowing for easy downloading, preprocessing, and training of models with various datasets.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/qlinhta/llm-psl.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Running with GPT-2 Model on AG News Dataset:

```bash
python3 ./src/run.py --train_file ./data/train.csv --test_file ./data/test.csv --epochs 5 --batch_size 8 --learning_rate 1e-5 --grad_accum_steps 4 --lora_dim 8```
```