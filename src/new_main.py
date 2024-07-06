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
import random
import os

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

def format_convert(read_file, write_file):
    with open(read_file, "r", encoding="utf8") as reader, \
         open(write_file, "w", encoding="utf8") as writer:
        for line in reader:
            items = line.strip().split("||")
            if len(items) < 2:
                logger.error(f"Line is not properly formatted: {line}")
                continue
            context = items[0]
            completion = items[1].strip("\n")
            x = {"context": context, "completion": completion}
            writer.write(json.dumps(x) + "\n")

def get_model_name(model_id):
    model_mapping = {
        1: "distilgpt2",
        2: "gpt2",
        3: "gpt2-medium",
        4: "gpt2-large",
        5: "gpt2-xl"
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


def encode (input, output, enc):
    writer = open(output, 'w')
    add_bos = True
    add_eos = True
    with open(input, 'r') as reader:
        line_idx = 0
        for line in reader:
            items = json.loads(line.strip())
            context = items['context']
            completion = items['completion']

            bos = 50256
            eos = 50256
            context_bpes= enc.encode(context)
            context_bpes += [bos] if add_bos else []

            completion_bpes = enc.encode(' ' + completion)
            completion_bpes += [eos] if add_eos else []

            ft_json = {}
            ft_json['context'] = context_bpes
            ft_json['completion'] = completion_bpes 
            writer.write(json.dumps(ft_json)+'\n')

            line_idx += 1
    writer.close()

def padding_tokens(tokens, max_seq_length, pad_token, direct, max_context_length=0):

    if max_context_length == 0:
        max_context_length = max_seq_length

    if len(tokens) > max_context_length:
        if direct > 0:
            pad_tokens = tokens[:max_context_length]
        else:
            pad_tokens = tokens[-max_context_length:]
    else:
        pad_tokens = tokens
    token_len = len(pad_tokens)
    pad_tokens = pad_tokens + [pad_token for _ in range(max_seq_length - token_len)]
    return pad_tokens, token_len


class FT_Dataset(Dataset):
    def __init__(self, ft_file, batch_size, max_seq_length, max_eval_length=0, joint_lm=False, prefix_len=0, infix_len=0, prefix_cursor=1000000, infix_cursor=2000000):
        self.ft_file = ft_file
        self.ft_samples = self.read_ft_file(ft_file)
        self.batch_size = batch_size
        self.num_examples = len(self.ft_samples)
        self.max_seq_length = max_seq_length
        self.max_eval_length = max_eval_length
        self.joint_lm = joint_lm
        self.prefix_len = prefix_len
        self.infix_len = infix_len
        self.prefix_cursor = prefix_cursor
        self.infix_cursor = infix_cursor

    def __len__(self):
        return self.num_examples

    def __getitem__(self, item):
        example = self.ft_samples[item]
        context = example[0]
        completion = example[1]
        pretokens = [i + self.prefix_cursor for i in range(0, self.prefix_len)]
        intokens = [i + self.infix_cursor for i in range(0, self.infix_len)]

        conditions = pretokens + context + intokens
        _input, _input_len = padding_tokens(conditions + completion, self.max_seq_length, 0, 1)

        pad_targets = [0 for i in range(0, self.prefix_len)] + context + [0 for i in range(0, self.infix_len)] + completion
        _target, _ = padding_tokens(pad_targets, self.max_seq_length, 0, 1)
        if not self.joint_lm:
            _msk = [0.0] * (len(conditions) - 1) + [1.0] * (_input_len - len(conditions))
        else:
            _msk = [1.0] * (_input_len - 1)

        _msk, _ = padding_tokens(_msk, self.max_seq_length, 0.0, 1)

        output = {}
        output["id"] = torch.tensor(item, dtype=torch.long)
        _query, _query_len = padding_tokens(
            conditions, self.max_seq_length, 0, -1,
            max_context_length=self.max_seq_length - self.max_eval_length
        )
        output["query"] = torch.tensor(_query, dtype=torch.long)
        output["query_len"] = torch.tensor(_query_len, dtype=torch.long)
        output["input"] = torch.tensor(_input, dtype=torch.long)
        output["target"] = torch.tensor(_target, dtype=torch.long)
        output["mask"] = torch.tensor(_msk, dtype=torch.float)
        return output

    def read_ft_file(self, ft_file):
        ft_samples = []
        with open(ft_file, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                context = items['context']
                completion = items['completion']
                ft_samples.append([context, completion])
        return ft_samples


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

        for idx, data in progress_bar:
            data = {key: value for key, value in data.items()}
            inputs = data['input'].to(device)
            labels = data['target'].to(device)
            masks = data['mask'].to(device)
            optimizer.zero_grad()
            with autocast(enabled=scaler is not None):
                outputs = model( input_ids=inputs,
                attention_mask=masks, labels=labels)
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
    format_convert("./data/train.txt", "./data/train_formatted.jsonl")
    format_convert("./data/test.txt", "./data/test_formatted.jsonl")

    model_name = get_model_name(args.model_id)
    logger.info(f"Selected model ID: {args.model_id}, Model Name: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    def read_json_file(file_path):
        with open(file_path, "r", encoding="utf8") as file:
            data = json.load(file)
            df = pd.DataFrame(data)
            return df
    encode('./data/train_formatted.jsonl', './data/train_formatted_token.jsonl', tokenizer)
    encode('./data/test_formatted.jsonl', './data/test_formatted_token.jsonl', tokenizer)
    train_data = FT_Dataset(
        ft_file ='./data/train_formatted_token.jsonl', batch_size=args.batch_size, max_seq_length=30, joint_lm= True)
    
  
    test_data = FT_Dataset(
        './data/test_formatted_token.jsonl', 1, 30,
    )
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=0,    
    )
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, num_workers=0, 
        shuffle=False, pin_memory=False, drop_last=False,   
    )
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
      data = {key: value for key, value in batch.items()}
      inputs = data['input'].to(device)
      targets = data['target'].to(device)
      masks = data['mask'].to(device)
      input_texts = [tokenizer.decode(inputs[i], skip_special_tokens=True) for i in
                       range(len(inputs))]
      targets= [tokenizer.decode(targets[i], skip_special_tokens=True) for i in
                       range(len(targets))]
      logger.info(f"Input: {input_texts[0]}")
      logger.info(f"Target: {targets[0]}")
      with torch.no_grad():
        input_len = int(torch.sum(masks).cpu().numpy())
        print(input_len)
        print(masks)
        output = lora_model.generate(inputs, max_new_tokens=30,
                                         attention_mask=masks,
                                         pad_token_id=tokenizer.eos_token_id,
                                         do_sample=False,
                                         temperature=0.9,
                                         top_k=40)
        logger.info(f"Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")

        pred_texts = [tokenizer.decode(o, skip_special_tokens=True).split('.')[0] for o in output]
      
        logger.info(f"Prediction: {pred_texts}")
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