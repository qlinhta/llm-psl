import torch
from tqdm import tqdm
import logging
from training.evaluation import compute_bleu, compute_chrf, compute_cider,compute_lrouge,compute_meteor,compute_rouge, compute_perplexity
from transformers import AutoTokenizer
def evaluate(model, dataloader, device, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.to(device)
    model.eval()
    eval_loss = 0.0

    preds = []
    labels_list = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            input_ids, labels, attention_mask =  batch
            inputs = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
  
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()
            print(logits.shape)
            predictions = torch.argmax(logits, dim=-1)
            preds.extend(predictions.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

            progress_bar.set_postfix(loss=eval_loss / (len(progress_bar)))

    print('***************************', preds[1])
    print('***************************', type(preds[0]))
    preds_text = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    labels_text = [tokenizer.decode(label, skip_special_tokens=True) for label in labels_list]

    print('--------------------- Predicted Texts:', preds_text)
    print('********************** Ground Truth Texts:', labels_text)
    bleu_score = compute_bleu(preds_text, labels_text)
    rouge_scores = compute_bleu(preds_text, labels_text)
    lrouge_scores = compute_lrouge(preds_text, labels_text)
    cider_scores = compute_cider(preds_text, labels_text)
    chrf_scores = compute_chrf(preds, labels, tokenizer)
    meteor_scores = compute_meteor(preds, labels, tokenizer)
    perplexity = compute_perplexity(eval_loss, len(dataloader))

    logging.info(f'Validation Loss: {eval_loss / len(dataloader):.4f}')
    logging.info(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    logging.info(f'BLEU Score: {bleu_score:.4f}')
    logging.info(f'ROUGE Scores: {rouge_scores}')
    logging.info(f'LROUGE Scores: {lrouge_scores}')
    logging.info(f'CIDER Scores: {cider_scores}')
    logging.info(f'METEOR Scores: {meteor_scores}')
    logging.info(f'Perplexity: {perplexity:.4f}')


    return eval_loss / len(dataloader)
