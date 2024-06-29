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
    elif id == 6:
        return preprocess_e2e(dataset)
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

def preprocess_e2e(dataset):
    def process(example):
        return {'text': example['meaning_representation'], 'labels': example['human_reference']}

    dataset = dataset.map(process, batched=True)
    return dataset
