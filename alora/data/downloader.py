import datasets

DATASET_OPTIONS = {
    1: "glue",
    2: "glue",
    3: "glue",
    4: "glue",
    5: "glue"
}

DATASET_NAMES = {
    1: "sst2",
    2: "mrpc",
    3: "rte",
    4: "cola",
    5: "stsb"
}


def download(id):
    if id not in DATASET_OPTIONS:
        raise ValueError(f"Dataset ID {id} is not defined.")

    dataset = datasets.load_dataset(DATASET_OPTIONS[id], DATASET_NAMES[id])
    return dataset
