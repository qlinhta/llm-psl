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
import sys
import io
import json
import pandas as pd
import string

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
        5: "gpt2-xl"
    }
    return model_mapping.get(model_id, "gpt2")


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


device = get_device()
logger.info(f"Using device: {device}")


class FineTuneDataset(Dataset):
    def __init__(self, data_file, batch_size, max_seq_length, sep_token_id, tokenizer, test=False):
        self.data_file = data_file
        self.samples = self._read_data_file(data_file)
        self.batch_size = batch_size
        self.num_samples = len(self.samples)
        self.max_seq_length = max_seq_length
        self.test = test
        self.sep_token_id = sep_token_id
        self.tokenizer = tokenizer

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        context, completion = self.samples[idx]
        sep_token_id = self.tokenizer.convert_tokens_to_ids("=")
        context_sep = context + [sep_token_id]
        input_tokens, input_len = self._pad_tokens(context_sep + completion, self.max_seq_length, 0, 1)
        input_tokens_test, input_len_test = self._pad_tokens(context, self.max_seq_length, 0, 1)
        target_tokens, _ = self._pad_tokens(context_sep[1:] + completion, self.max_seq_length, 0, 1)

        mask = [0.0] * (len(context_sep) - 1) + [1.0] * (input_len - len(context_sep)) if not self.test else [1.0] * (
                    input_len - 1)
        mask, _ = self._pad_tokens(mask, self.max_seq_length, 0.0, 1)
        if self.test:
            input_tokens_test = context_sep
            target_tokens_test = completion
            return {
                "id": torch.tensor(idx, dtype=torch.long),
                "input": torch.tensor(input_tokens_test, dtype=torch.long),
                "target": torch.tensor(target_tokens_test, dtype=torch.long),
                # "mask": torch.tensor(mask, dtype=torch.float)
            }
        else:
            return {
                "id": torch.tensor(idx, dtype=torch.long),
                "input": torch.tensor(input_tokens, dtype=torch.long),
                "target": torch.tensor(target_tokens, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.float)
            }

    """
    def _read_data_file(self, data_file):
          with open(data_file, 'r') as file:
              return [[data['context'], data['completion']] for data in map(json.loads, file)]
    """

    def _read_data_file(self, data_file):
        with open(data_file, 'r') as file:
            data_list = []
            for i, line in enumerate(file):

                if i >= 1000:  # Stop after reading 10 lines
                    break

                data = json.loads(line)
                data_list.append([data['context'], data['completion']])

            return data_list

    def _pad_tokens(self, tokens, max_len, pad_token, direction, max_context_len=0):
        max_context_len = max_context_len or max_len
        if len(tokens) > max_context_len:
            tokens = tokens[:max_context_len] if direction > 0 else tokens[-max_context_len:]
        tokens += [pad_token] * (max_len - len(tokens))
        return tokens, len(tokens)


class LoRALinearLayer(nn.Module):
    def __init__(self, weight, bias, lora_dim):
        super(LoRALinearLayer, self).__init__()
        self.linear = nn.Linear(weight.shape[1], weight.shape[0], bias=(bias is not None))
        self.linear.weight = nn.Parameter(weight)
        if bias is not None:
            self.linear.bias = nn.Parameter(bias)
        self.lora_right = nn.Parameter(torch.zeros(weight.shape[1], lora_dim))
        self.lora_left = nn.Parameter(torch.zeros(lora_dim, weight.shape[0]))
        nn.init.kaiming_uniform_(self.lora_right, a=math.sqrt(5))

    def forward(self, x):
        frozen_output = self.linear(x)
        lora_output = x @ self.lora_right @ self.lora_left
        return frozen_output + lora_output


def integrate_lora_layers(model, lora_dim=8):
    count = 0
    for name, module in model.named_modules():
        if "attn.c_attn" in name:
            lora_layer = LoRALinearLayer(
                weight=module.weight.t(),
                bias=module.bias,
                lora_dim=lora_dim
            ).to(device)
            parent_name = ".".join(name.split(".")[:-1])
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, name.split(".")[-1], lora_layer)
            count += 1
    logger.info(f"Integrated LoRA layers with dimension: {lora_dim} in {count} layers")


def freeze_model_parameters(model):
    logger.info("Freezing model parameters except for LoRA layers")
    for name, param in model.named_parameters():
        param.requires_grad = "lora_right" in name or "lora_left" in name


def display_model_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    table = PrettyTable(["Parameter Type", "Count"])
    table.add_row(["Trainable Params", trainable_params])
    table.add_row(["Total Params", total_params])
    table.add_row(["Trainable %", f"{100 * trainable_params / total_params:.2f}%"])
    logger.info(f"\n{table}")


def train_model(model, dataloader, epochs=1, batch_size=8, lr=1e-5, grad_accum_steps=4):
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler() if device.type == 'cuda' else None
    model.train()
    losses = []

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs} started with learning rate: {lr}")
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")

        for idx, batch in progress_bar:
            inputs, labels, masks = batch['input'].to(device), batch['target'].to(device), batch['mask'].to(device)
            optimizer.zero_grad()

            with autocast(enabled=scaler is not None):
                outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
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

            progress_bar.set_postfix({"Loss": running_loss / (idx + 1)})
        losses.append(running_loss / len(dataloader))
        logger.info(f"Epoch {epoch + 1}/{epochs} completed with average loss: {running_loss / len(dataloader):.4f}")

    return losses


def plot_losses(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    plt.show()


def evaluate_model(model, dataloader, tokenizer, sep_token="="):
    model.eval()
    all_predictions, all_references = [], []
    punctuation_table = str.maketrans('', '', string.punctuation)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # inputs, masks, labels = batch['input'].to(device),batch['mask'].to(device), batch['target'].to(device)
            inputs, labels = batch['input'].to(device), batch['target'].to(device)
            generated_outputs = model.generate(inputs, max_new_tokens=30,
                                               pad_token_id=tokenizer.eos_token_id,
                                               do_sample=False,
                                               temperature=0.9,
                                               top_k=40
                                               )
            predictions = tokenizer.batch_decode(generated_outputs, clean_up_tokenization_spaces=True,
                                                 keep_special_token=True)
            references = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True, keep_special_token=True)

            # Only take the completion part (after separator token) for evaluation
            references = [ref.split(sep_token)[-1] for ref in references]
            predictions = [pred.split(sep_token)[-1].split('\n')[0] + "<|endoftext|>" for pred in predictions]
            # print('predictionnnnns', predictions)
            # print('references', references)
            # predictions = [pred.translate(punctuation_table) for pred in predictions]
            # references = [ref.translate(punctuation_table) for ref in references]
            # predictions =[pred.lstrip('\n').split('\n')[0] for pred in predictions]
            all_predictions.extend(predictions)
            all_references.extend(references)

    bleu_score = compute_bleu(all_references, all_predictions)
    meteor_score = compute_meteor(all_references, all_predictions)
    rouge_l_score = compute_rouge_l(all_references, all_predictions)
    cider_score = compute_cider(all_references, all_predictions)
    chrf_score = compute_chrf(all_references, all_predictions)

    evaluation_scores = {
        "BLEU": bleu_score,
        "METEOR": meteor_score,
        "ROUGE_L": rouge_l_score,
        "CIDER": cider_score,
        "CHRF": chrf_score
    }
    output_file_path = 'predictions.txt'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for prediction in all_predictions:
            f.write(prediction + '\n')

    print(f"Predictions saved to {output_file_path}")

    return evaluation_scores


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a GPT model with LoRA.")
    parser.add_argument("--model_id", type=int, choices=[1, 2, 3, 4, 5], default=2, help="Choose a GPT model.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--max_seq_length", type=int, default=60, help="Maximum sequence length for input data.")
    parser.add_argument("--lora_dim", type=int, default=8, help="LoRA dimension.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps.")
    args = parser.parse_args()

    model_name = get_model_name(args.model_id)
    logger.info(f"Selected model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    train_file = './data/train_formatted_token.jsonl'
    test_file = './data/test_formatted_token.jsonl'
    sep_token = "<|endoftext|>"
    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
    if sep_token_id is None:
        raise ValueError(f"The separator token '{sep_token}' is not found in the tokenizer's vocabulary.")
    train_dataset = FineTuneDataset(data_file=train_file, batch_size=args.batch_size,
                                    max_seq_length=args.max_seq_length, test=False, sep_token_id=sep_token_id,
                                    tokenizer=tokenizer)
    test_dataset = FineTuneDataset(data_file=test_file, batch_size=1, max_seq_length=args.max_seq_length, test=True,
                                   sep_token_id=sep_token_id, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    integrate_lora_layers(model, args.lora_dim)
    freeze_model_parameters(model)
    display_model_parameters(model)

    training_losses = train_model(model, train_dataloader, args.epochs, args.batch_size, args.learning_rate,
                                  args.grad_accum_steps)
    plot_losses(training_losses)

    eval_scores = evaluate_model(model, test_dataloader, tokenizer)
    logger.info(f"Evaluation Scores: {eval_scores}")


if __name__ == "__main__":
    main()
