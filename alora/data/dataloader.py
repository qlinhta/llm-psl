from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def tokenize(dataset, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def tokenize_function(example):
        return tokenizer(example['text'], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def create_dataloader(dataset, batch_size=32):
    subset = dataset.select(range(1000))
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    subset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return dataloader
