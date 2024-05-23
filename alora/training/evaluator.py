import torch
from tqdm import tqdm
import logging
from training.evaluation import compute_metrics, compute_bleu, compute_rouge, compute_perplexity


def evaluate(model, dataloader, device, tokenizer):
    model.to(device)
    model.eval()
    eval_loss = 0.0

    preds = []
    labels_list = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()

            preds.extend(torch.argmax(logits, dim=-1).tolist())
            labels_list.extend(labels.tolist())

            progress_bar.set_postfix(loss=eval_loss / (len(progress_bar)))

    accuracy, precision, recall, f1 = compute_metrics(preds, labels_list)
    bleu_score = compute_bleu(preds, labels_list, tokenizer)
    rouge_scores = compute_rouge(preds, labels_list, tokenizer)
    perplexity = compute_perplexity(eval_loss, len(dataloader))

    logging.info(f'Validation Loss: {eval_loss / len(dataloader):.4f}')
    logging.info(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    logging.info(f'BLEU Score: {bleu_score:.4f}')
    logging.info(f'ROUGE Scores: {rouge_scores}')
    logging.info(f'Perplexity: {perplexity:.4f}')

    return eval_loss / len(dataloader)
