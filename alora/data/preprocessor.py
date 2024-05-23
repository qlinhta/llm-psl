"""
LoRA_Project/
│
├── data/
│   ├── __init__.py
│   ├── dataset_downloader.py
│   ├── dataset_preprocessor.py
│   └── dataloader.py
│
├── models/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── models.py
│   └── small_llms/
│       ├── model1.py
│       ├── model2.py
│       └── ...
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── evaluation.py  # New file for evaluation metrics
│
├── utils/
│   ├── __init__.py
│   ├── config.py
│   └── logger.py
│
├── main.py
├── requirements.txt
└── README.md

"""


def preprocess(dataset, id):
    if id == 1:
        return preprocess_sst2(dataset)
    elif id == 2:
        return preprocess_mrpc(dataset)
    elif id == 3:
        return preprocess_rte(dataset)
    elif id == 4:
        return preprocess_cola(dataset)
    elif id == 5:
        return preprocess_stsb(dataset)
    else:
        raise ValueError(f"Dataset ID {id} is not defined.")


def preprocess_sst2(dataset):
    def process(example):
        return {'text': example['sentence'], 'labels': example['label']}

    dataset = dataset.map(process, batched=True)
    return dataset


def preprocess_mrpc(dataset):
    def process(example):
        return {'text': example['sentence1'] + ' [SEP] ' + example['sentence2'], 'labels': example['label']}

    dataset = dataset.map(process, batched=True)
    return dataset


def preprocess_rte(dataset):
    def process(example):
        return {'text': example['sentence1'] + ' [SEP] ' + example['sentence2'], 'labels': example['label']}

    dataset = dataset.map(process, batched=True)
    return dataset


def preprocess_cola(dataset):
    def process(example):
        return {'text': example['sentence'], 'labels': example['label']}

    dataset = dataset.map(process, batched=True)
    return dataset


def preprocess_stsb(dataset):
    def process(example):
        return {'text': example['sentence1'] + ' [SEP] ' + example['sentence2'], 'labels': example['label']}

    dataset = dataset.map(process, batched=True)
    return dataset
