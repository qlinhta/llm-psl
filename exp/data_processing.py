from datasets import load_dataset
from transformers import GPT2Tokenizer, DistilBERTTokenizer


def load_and_preprocess_data(dataset_id, model_name):
    if dataset_id == 'movie_dialogues':
        dataset = load_dataset('cornell_movie_dialogs')
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        def preprocess(example):
            return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

        dataset = dataset.map(preprocess, batched=True)

    elif dataset_id == 'ag_news':
        dataset = load_dataset('ag_news')
        tokenizer = DistilBERTTokenizer.from_pretrained(model_name)

        def preprocess(example):
            return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

        dataset = dataset.map(preprocess, batched=True)

    elif dataset_id == 'imdb':
        dataset = load_dataset('imdb')
        tokenizer = DistilBERTTokenizer.from_pretrained(model_name)

        def preprocess(example):
            return tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)

        dataset = dataset.map(preprocess, batched=True)

    else:
        raise ValueError("Unsupported dataset_id")

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return dataset
