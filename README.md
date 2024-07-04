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

**Model IDs:**

| Model ID | Model Name       | # Parameters |
|----------|------------------|--------------|
| 1        | GPT-2            | 117M         |
| 2        | GPT-Medium       | 345M         |
| 3        | GPT-Large        | 762M         |
| 4        | GPT-XL           | 1.5B         |
| 5        | DistilGPT-2      | 82M          |
| 6        | Meta-Llama-3-8B  | 8B           |
| 7        | Meta-Llama-3-70B | 70B          |
| 8        | Gemma-2-9B       | 9B           |
| 9        | Gemma-2-27B      | 27B          |

Running with GPT-2 Model on AG News Dataset:

```bash
python3 ./src/run.py --model_id 1 --train_file ./data/train.csv --test_file ./data/test.csv --epochs 5 --batch_size 8 --learning_rate 1e-5 --grad_accum_steps 4 --lora_dim 8
```