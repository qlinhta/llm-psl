from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch


def tokenize(dataset, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def tokenize_function(example):
        return tokenizer(example['text'], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


# def create_dataloader(dataset, batch_size=32):
#     dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return dataloader

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    fast_tokenizer=True)
block_size = 512
device = torch.device("cuda")

def fill_ignore_label(l, c):
    l[:len(c) - 1] = [-100] * (len(c) - 1)
    return l

def pad_tokens(tokens, max_seq_length, padding_token):
    res_tokens = tokens[:max_seq_length]
    token_len = len(res_tokens)
    res_tokens = res_tokens + \
        [padding_token for _ in range(max_seq_length - token_len)]
    return res_tokens

def collate_batch(batch):
    # tokenize both context and completion respectively
    # (context and completion is delimited by "\n")
    context_list = list(zip(*batch))[0]

    context_list = [c + "\n" for c in context_list]
    completion_list = list(zip(*batch))[1]
    context_result = tokenizer(context_list)
    context_tokens = context_result["input_ids"]
    context_masks = context_result["attention_mask"]
    completion_result = tokenizer(completion_list)
    completion_tokens = completion_result["input_ids"]
    completion_masks = completion_result["attention_mask"]
    # concatenate token
    inputs = [i + j for i, j in zip(context_tokens, completion_tokens)]
    masks = [i + j for i, j in zip(context_masks, completion_masks)]
    # create label
    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
    labels = [t[1:] + [eos_id] for t in inputs]
    labels = list(map(fill_ignore_label, labels, context_tokens))
    # truncate and pad tokens
    inputs = [pad_tokens(t, block_size, 0) for t in inputs] # OPT and GPT-2 doesn't use pad token (instead attn mask is used)
    masks = [pad_tokens(t, block_size, 0) for t in masks]
    labels = [pad_tokens(t, block_size, -100) for t in labels]
    # convert to tensor
    inputs = torch.tensor(inputs, dtype=torch.int64).to(device)
    masks = torch.tensor(masks, dtype=torch.int64).to(device)
    labels = torch.tensor(labels, dtype=torch.int64).to(device)
    # inputs = torch.tensor(inputs, dtype=torch.int64)
    # masks = torch.tensor(masks, dtype=torch.int64)
    # labels = torch.tensor(labels, dtype=torch.int64)
    # dico = {"input_ids": inputs, "attention_mask":masks, "labels":labels}
    
    return inputs, labels, masks
    # return dico

def create_dataloader(dataset, batch_size=32):
    
    small_dataset = dataset.select(range(1000))
    dataloader = DataLoader(
    small_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
    )
    return dataloader