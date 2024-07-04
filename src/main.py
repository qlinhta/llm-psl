import os
import argparse
import numpy as np
import math
import warnings
from cryptography.fernet import Fernet
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from prettytable import PrettyTable
from torch.cuda.amp import GradScaler, autocast
import logging
from colorlog import ColoredFormatter
from huggingface_hub import login
import matplotlib.pyplot as plt
from evaluation import compute_bleu, compute_meteor, compute_rouge_l, compute_cider, compute_chrf

warnings.filterwarnings("ignore")

plt.style.use('default')
plt.rc('font', family='sans-serif', size=14)
plt.rc('axes', titlesize=14, labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=14)
plt.rc('lines', markersize=7)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open('secret.key', 'rb') as key_file:
    key = key_file.read()
with open('encrypted_token.txt', 'rb') as token_file:
    encrypted_token = token_file.read()
cipher = Fernet(key)
token = cipher.decrypt(encrypted_token).decode()

login(token=token)


def setup_logger(name):
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'orange'}
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
        1: "distilgpt2",
        2: "gpt2",
        3: "gpt2-medium",
        4: "gpt2-large",
        5: "gpt2-xl",
        6: "meta-llama/Meta-Llama-3-8B",
        7: "meta-llama/Meta-Llama-3-70B",
        8: "google/gemma-2-9b",
        9: "google/gemma-2-27b",
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


def collate_batch(batch, tokenizer, block_size):
    texts = [item['text'] for item in batch]
    targets = [item['target'] for item in batch]
    tokens = tokenizer(texts, padding='max_length', truncation=True, max_length=block_size, return_tensors="pt")
    tokens = {key: val.to(device) for key, val in tokens.items()}
    labels = tokens["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return tokens, labels, targets


def create_dataloader(dataset, tokenizer, batch_size=32, block_size=512):
    def collate_fn(batch):
        return collate_batch(batch, tokenizer, block_size)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


class CustomDataset(Dataset):
    def __init__(self, file_path, is_train=True):
        self.data = self.load_data(file_path, is_train)

    def load_data(self, file_path, is_train):
        dataset = []
        with open(file_path, 'r') as file:
            for line in file:
                attributes, description = line.strip().split('||')
                attribute_dict = {}
                for attribute in attributes.split('|'):
                    key, value = attribute.split(':', 1)
                    attribute_dict[key.strip()] = value.strip()
                if is_train:
                    text = f"{attribute_dict} Description: {description.strip()}"
                else:
                    text = f"{attribute_dict} Description:"
                target = description.strip() if not is_train else None
                dataset.append({'attributes': attribute_dict, 'text': text, 'target': target})
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
    number_of_lora_layers = 0
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
            number_of_lora_layers += 1
    logger.info(f"Integrated LoRA layers with dimension: {lora_dim} in {number_of_lora_layers} layers")


def freeze(model: nn.Module) -> None:
    logger.info("Freezing the model weights except for the LoRA layers")
    for n, p in model.named_parameters():
        p.requires_grad = "lora_right" in n or "lora_left" in n


def params(model: nn.Module) -> None:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    table = PrettyTable(["Parameter Type", "Count"])
    table.add_row(["Trainable Params", trainable_params])
    table.add_row(["All Params", all_params])
    table.add_row(["Trainable %", f"{100 * trainable_params / all_params:.2f}"])
    logger.info(f"\n{table}")


def train(model: nn.Module, dataloader: DataLoader, epochs: int = 1, batch_size: int = 8, learning_rate: float = 1e-5,
          grad_accum_steps: int = 4) -> list:
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler() if device.type == 'cuda' else None
    model.train()
    losses = []

    for epoch in range(epochs):
        logger.info(f"EPOCH: {epoch + 1}/{epochs} started with learning rate: {learning_rate}")
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")

        for idx, (inputs, labels, _) in progress_bar:
            optimizer.zero_grad()
            with autocast(enabled=scaler is not None):
                outputs = model(**inputs, labels=labels)
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


def main(args) -> None:
    global tokenizer

    table = PrettyTable()
    table.field_names = ["ID", "Model Name"]
    for i in range(1, 10):
        table.add_row([i, get_model_name(i)])
    logger.info(f"\n{table}")

    model_name = get_model_name(args.model_id)
    logger.info(f"Selected model ID: {args.model_id}, Model Name: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    train_data = CustomDataset('./data/train.txt', is_train=True)
    test_data = CustomDataset('./data/val.txt', is_train=False)

    if args.train_limit:
        train_data.data = train_data.data[:args.train_limit]
    if args.eval_limit:
        test_data.data = test_data.data[:args.eval_limit]

    train_dataloader = create_dataloader(train_data, tokenizer, batch_size=args.batch_size)
    test_dataloader = create_dataloader(test_data, tokenizer, batch_size=args.batch_size)

    logger.info(f"Attempting to load model: {model_name}")
    lora_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    integrate(lora_model, lora_dim=args.lora_dim)
    freeze(lora_model)
    params(lora_model)
    losses = train(lora_model, train_dataloader, epochs=args.epochs, batch_size=args.batch_size,
                   learning_rate=args.learning_rate, grad_accum_steps=args.grad_accum_steps)

    logger.info(f"Evaluating the model: {len(test_dataloader)} samples")
    preds, labels = [], []
    progress_bar = tqdm(total=len(test_dataloader), desc="Evaluating")
    for batch in test_dataloader:
        inputs, _, targets = batch
        input_texts = [tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True) for i in
                       range(len(inputs['input_ids']))]
        logger.info(f"Input: {input_texts[0]}")
        logger.info(f"Target: {targets[0]}")
        with torch.no_grad():
            output = lora_model.generate(inputs['input_ids'], max_new_tokens=60,
                                         attention_mask=inputs['attention_mask'],
                                         pad_token_id=tokenizer.eos_token_id,
                                         do_sample=False,
                                         temperature=0.9,
                                         top_k=40)
        # logger.info(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
        pred_texts = [tokenizer.decode(o, skip_special_tokens=True).split("Description:")[-1].strip() for o in output]
        logger.info(f"Prediction: {pred_texts[0]}")
        preds.extend(pred_texts)
        labels.extend(targets)
        progress_bar.update(1)
    progress_bar.close()

    bleu = compute_bleu(preds, labels)
    meteor = compute_meteor(preds, labels)
    rouge_l = compute_rouge_l(preds, labels)
    cider = compute_cider(preds, labels)
    chrf = compute_chrf(preds, labels)

    table = PrettyTable()
    table.field_names = ["Metric", "Score"]
    table.add_row(["BLEU", f"{bleu}"])
    table.add_row(["METEOR", f"{meteor}"])
    table.add_row(["ROUGE-L", f"{rouge_l}"])
    table.add_row(["CIDEr", f"{cider}"])
    table.add_row(["CHRF", f"{chrf}"])
    logger.info(f"\n{table}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(losses, marker='o', linestyle='-', color='b', label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(
        f"{model_name}: BLEU: {bleu:.2f}, METEOR: {meteor:.2f}, ROUGE-L: {rouge_l:.2f}, CIDEr: {cider:.2f}, CHRF: {chrf:.2f}",
        fontsize=10)
    ax.legend()
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.2', color='grey')
    plt.grid(which='minor', linestyle=':', linewidth='0.2', color='grey')
    fig.savefig(f"./figures/{model_name}_{args.lora_dim}_{args.epochs}_{args.batch_size}_{args.learning_rate}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model with LoRA")
    parser.add_argument("--model_id", type=int, default=1, help="Identifier for the model to use")
    parser.add_argument("--train_limit", type=int, default=None, help="Limit the number of samples")
    parser.add_argument("--eval_limit", type=int, default=None, help="Limit the number of samples for evaluation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lora_dim", type=int, default=8, help="Dimension of the LoRA layers")
    args = parser.parse_args()
    main(args)
