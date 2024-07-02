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

def generate_text(model, input, mask, eos_id, pred_sequence_length):
    predicted_last_id = -1
    start_token_len = torch.sum(mask).cpu().numpy()
    token_len = start_token_len
    with torch.no_grad():
        while (predicted_last_id != eos_id) and \
              (token_len - start_token_len < pred_sequence_length):
            output = model(
                input_ids=input,
                attention_mask=mask,
            )
            predicted_ids = torch.argmax(output.logits, axis=-1).cpu().numpy()
            predicted_last_id = predicted_ids[0][token_len - 1]
            input[0][token_len] = predicted_last_id
            mask[0][token_len] = 1
            token_len = torch.sum(mask).cpu().numpy()
    return input, token_len

def main():
    args = parse_args()
    logger = setup_logger('LoRA', 'training.log')

    dataset = download(args.dataset)
    dataset = preprocess(dataset, args.dataset)
    #tokenized_dataset = tokenize(dataset['train'], args.tokenizer)
    # train_dataloader = create_dataloader(dataset['train'], batch_size=args.batch_size)

    # model = load(args.model)
    model = apply(model, rank=args.rank)

    device = __device__()

    # model = train(model, train_dataloader, epochs=args.epochs, learning_rate=args.learning_rate, device=device)

    #val_tokenized_dataset = tokenize(dataset['validation'], args.tokenizer)
    val_dataloader = create_dataloader(dataset['validation'], batch_size=args.batch_size)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    fast_tokenizer=True)
    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
    for i, (input, _, mask) in enumerate(val_dataloader):
        if i == 5:
            break
        print("********** input **********")
        input_len = torch.sum(mask).cpu().numpy()
        print(tokenizer.decode(input[0][:input_len]))
        result_token, result_len = generate_text(
            model,
            input,
            mask,
            eos_id,
            pred_sequence_length=30)
        print("********** result **********")
        print(tokenizer.decode(result_token[0][:result_len]))



    
    eval_loss = evaluate(model, val_dataloader, device=device, tokenizer_name=args.tokenizer)
    logger.info(f"Final Evaluation - Loss: {eval_loss:.4f}")


if __name__ == "__main__":
    main()
