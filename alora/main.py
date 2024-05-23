from data.downloader import download
from data.preprocessor import preprocess
from data.dataloader import tokenize, create_dataloader
from models.loader import load
from models.lora import apply
from training.trainer import train
from training.evaluator import evaluate
from utils.config import parse_args
from utils.logger import setup_logger
import torch


def __device__() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    args = parse_args()
    logger = setup_logger('LoRA', 'training.log')

    dataset = download(args.dataset)
    dataset = preprocess(dataset, args.dataset)
    tokenized_dataset = tokenize(dataset['train'], args.tokenizer)
    train_dataloader = create_dataloader(tokenized_dataset, batch_size=args.batch_size)

    model = load(args.model)
    model = apply(model, rank=args.rank)

    device = __device__()

    model = train(model, train_dataloader, epochs=args.epochs, learning_rate=args.learning_rate, device=device)

    val_tokenized_dataset = tokenize(dataset['validation'], args.tokenizer)
    val_dataloader = create_dataloader(val_tokenized_dataset, batch_size=args.batch_size)
    eval_loss = evaluate(model, val_dataloader, device=device, tokenizer=args.tokenizer)
    logger.info(f"Final Evaluation - Loss: {eval_loss:.4f}")


if __name__ == "__main__":
    main()
