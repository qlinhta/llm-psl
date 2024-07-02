import argparse
from transformers import GPT2LMHeadModel, DistilBERTForSequenceClassification, Trainer, TrainingArguments
from data_processing import load_and_preprocess_data


def main():
    parser = argparse.ArgumentParser(description="LoRA Experiment")
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name')
    parser.add_argument('--dataset_id', type=str, required=True, help='Dataset identifier')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA')
    parser.add_argument('--use_custom_lora', action='store_true', help='Use custom LoRA implementation')

    args = parser.parse_args()

    dataset = load_and_preprocess_data(args.dataset_id, args.model_name)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    if 'gpt2' in args.model_name:
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
    elif 'distilbert' in args.model_name:
        model = DistilBERTForSequenceClassification.from_pretrained(args.model_name,
                                                                    num_labels=4 if args.dataset_id == 'ag_news' else 2)
    else:
        raise ValueError("Unsupported model")

    if args.use_custom_lora:
        from alora import ALoRA
        model = ALoRA(model, rank=args.lora_rank)
    else:
        from lora import LoRA
        model = LoRA(model, rank=args.lora_rank)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
