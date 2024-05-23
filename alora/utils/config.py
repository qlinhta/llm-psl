import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Training")
    parser.add_argument("--dataset", type=int, required=True, help="ID of the dataset")
    parser.add_argument("--model", type=str, required=True, help="Name of the model")
    parser.add_argument("--tokenizer", type=str, required=True, help="Name of the tokenizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--rank", type=int, default=4, help="Rank for LoRA")
    return parser.parse_args()
