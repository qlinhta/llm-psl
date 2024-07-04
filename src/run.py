import os
import argparse
import time
import numpy as np
import pandas as pd
import math
import warnings

from nltk import choose

warnings.filterwarnings("ignore")

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from prettytable import PrettyTable
from torch.cuda.amp import GradScaler, autocast
import logging
from colorlog import ColoredFormatter
from datasets import load_dataset
import matplotlib.pyplot as plt
from evaluation import compute_bleu, compute_rouge, compute_chrf, compute_perplexity

plt.style.use('default')
plt.rc('text', usetex=False)
plt.rc('font', family='sans-serif')
plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=14)
plt.rc('lines', markersize=7)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logger(name):
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        },
        secondary_log_colors={},
        style='%'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


logger = setup_logger(__name__)


def get_model_name(model_id):
    model_mapping = {
        1: "gpt2",
        2: "gpt2-medium",
        3: "gpt2-large",
        4: "gpt2-xl",
        5: "distilgpt2"
    }
    return model_mapping.get(model_id, "gpt2")


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


device = device()
logger.info(f"Using device: {device}")


class TextDataset(Dataset):
    def __init__(self, texts: list) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def params(model: nn.Module) -> None:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    table = PrettyTable(["Parameter Type", "Count"])
    table.add_row(["Trainable Params", trainable_params])
    table.add_row(["All Params", all_params])
    table.add_row(["Trainable %", f"{100 * trainable_params / all_params:.2f}"])
    logger.info(f"\n{table}")


def train(model: nn.Module, dataset: Dataset, epochs: int = 1, batch_size: int = 8, learning_rate: float = 1e-5,
          grad_accum_steps: int = 4) -> list:
    global avg_loss
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler() if device.type == 'cuda' else None
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

    for epoch in range(epochs):
        logger.info(f"EPOCH: {epoch + 1}/{epochs} started with learning rate: {learning_rate}")
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        for idx, batch in progress_bar:
            optimizer.zero_grad()
            batch = [str(text) for text in batch]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss / grad_accum_steps
            running_loss += loss.item()

            if scaler:
                scaler.scale(loss).backward()
                if (idx + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (idx + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            avg_loss = running_loss / (idx + 1)
            progress_bar.set_postfix({'Loss': f"{loss.item() * grad_accum_steps:.4f}", 'Avg Loss': f"{avg_loss:.4f}"})
        losses.append(avg_loss)

    return losses


class LoRALinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, lora_dim: int) -> None:
        super(LoRALinear, self).__init__()
        out, inp = weight.shape
        self.linear = nn.Linear(inp, out, bias=bias is not None)
        self.linear.weight = nn.Parameter(weight)
        if bias is not None:
            self.linear.bias = nn.Parameter(bias)
        self.lora_right = nn.Parameter(torch.zeros(inp, lora_dim))
        nn.init.kaiming_uniform_(self.lora_right, a=math.sqrt(5))
        self.lora_left = nn.Parameter(torch.zeros(lora_dim, out))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        frozen_output = self.linear(input)
        lora_output = input @ self.lora_right.to(input.device) @ self.lora_left.to(input.device)
        return frozen_output + lora_output


def integrate(model: nn.Module, lora_dim: int = 8) -> None:
    for name, module in model.named_modules():
        if "attn.c_attn" in name:
            lora_layer = LoRALinear(
                weight=torch.transpose(module.weight, 0, 1).to(device),
                bias=module.bias.to(device) if module.bias is not None else None,
                lora_dim=lora_dim
            ).to(device)
            parent_name = ".".join(name.split(".")[:-1])
            parent_module = model.get_submodule(parent_name)
            parent_module.__setattr__(name.split(".")[-1], lora_layer)
    logger.info(f"Integrated LoRA layers with dimension: {lora_dim}")


def freeze(model: nn.Module) -> None:
    logger.info("Freezing the model weights except for the LoRA layers")
    for n, p in model.named_parameters():
        p.requires_grad = "lora_right" in n or "lora_left" in n


def generate(model: nn.Module, query: str, max_length: int = 100, top_n: int = 1) -> None:
    inputs = tokenizer(query, return_tensors="pt", padding=True).to(device)
    cur_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    for _ in range(max_length):
        outputs = model(cur_ids, attention_mask=attention_mask)
        logits = outputs.logits
        softmax_logits = torch.softmax(logits[0, -1], dim=0)
        next_token_id = choose(softmax_logits.to('cpu').detach().numpy(), n=top_n)
        cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1)).long().to(device)], dim=1)
        if next_token_id in tokenizer.encode(''):
            break
        print(tokenizer.decode([next_token_id]), end='')


def main(args) -> None:
    global tokenizer
    model_name = get_model_name(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = "<PAD>"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)

    train_texts = train_df['Description'][:100].tolist()
    test_texts = test_df['Description'][:10].tolist()

    train_dataset = TextDataset(train_texts)
    test_dataset = TextDataset(test_texts)

    lora_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    integrate(lora_model, lora_dim=args.lora_dim)
    freeze(lora_model)
    params(lora_model)
    losses = train(lora_model, train_dataset, epochs=args.epochs, batch_size=args.batch_size,
                   learning_rate=args.learning_rate,
                   grad_accum_steps=args.grad_accum_steps)

    logger.info(f"Evaluating the model: {len(test_dataset)} samples")
    preds, labels = [], []
    progress_bar = tqdm(total=len(test_dataset), desc="Evaluating")
    for i in range(len(test_dataset)):
        input_text = test_dataset[i]
        inputs = tokenizer(input_text, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            output = lora_model.generate(input_ids, max_new_tokens=50, attention_mask=attention_mask,
                                         pad_token_id=tokenizer.eos_token_id)
        pred_text = tokenizer.decode(output[0], skip_special_tokens=True)
        label_text = input_text
        preds.append(pred_text)
        labels.append(label_text)
        progress_bar.update(1)
    progress_bar.close()

    bleu = compute_bleu(preds, labels)
    rouge = compute_rouge(preds, labels)
    chrf = compute_chrf(preds, labels)

    logger.info(f"BLEU Score: {bleu}")
    logger.info(f"ROUGE Scores: {rouge}")
    logger.info(f"CHRF Score: {chrf}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(losses, marker='o', linestyle='-', color='b', label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f"{model_name}: BLUE: {bleu:.2f}, ROUGE: {rouge['rouge-l']['f']:.2f}, CHRF: {chrf:.2f}")
    ax.legend()
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.2', color='grey')
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='grey')
    fig.savefig(f"./figures/{time.time()}_loss_{model_name}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model with LoRA")
    parser.add_argument("--model_id", type=int, default=1, help="Identifier for the model to use")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the testing dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lora_dim", type=int, default=8, help="Dimension of the LoRA layers")
    args = parser.parse_args()
    main(args)
