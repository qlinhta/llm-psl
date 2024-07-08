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
| 1        | DistilGPT2       | 82M          |
| 2        | GPT2             | 124M         |
| 3        | GPT2-Medium      | 345M         |
| 4        | GPT2-Large       | 774M         |
| 5        | GPT2-XL          | 1558M        |
| 6        | Meta-Llama-3-8B  | 8B           |
| 7        | Meta-Llama-3-70B | 70B          |
| 8        | Gemma-2-9B       | 9B           |
| 9        | Gemma-2-27B      | 27B          |

Running with Models on E2E Dataset:

```bash
python3 ./src/run.py --train_limit [N] --eval_limit [N] --model_id [id] --epochs [ep] --batch_size [b] --learning_rate [lr] --grad_accum_steps [grad] --lora_dim [dim]```