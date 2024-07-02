import torch
from tqdm import tqdm
import logging
from training.evaluation import compute_bleu, compute_chrf, compute_cider,compute_lrouge,compute_meteor,compute_rouge, compute_perplexity
from transformers import AutoTokenizer

def generate_text(model, input, mask, eos_id, pred_sequence_length, labels, tokenizer):
    input = input.unsqueeze(0)
    mask = mask.unsqueeze(0)
    print('inpuuuuut_shape', input.shape)
    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
    predicted_last_id = -1
    start_token_len = torch.sum(mask).cpu().numpy()
    token_len = start_token_len

    with torch.no_grad():
        while (predicted_last_id != eos_id) and \
              (token_len - start_token_len < pred_sequence_length):
            output = model(
                input_ids=input,
                attention_mask=mask,
                labels=labels
            )

            loss = output.loss
            # eval_loss += loss.item()
            predicted_ids = torch.argmax(output.logits, axis=-1).cpu().numpy()
            print('predicted_ids', predicted_ids.shape)
            predicted_last_id = predicted_ids[0][token_len - 1]
            input[0][token_len] = predicted_last_id
            mask[0][token_len] = 1
            token_len = torch.sum(mask).cpu().numpy()
    print('ennnnnnnnnnnnnnnnnd shape', input.shape)
    return input[:][start_token_len-1:], loss


def evaluate(model, dataloader, device, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_id = tokenizer.encode(tokenizer.eos_token)[0]
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
            # outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            preds_one = []
            for i, in_sentence in enumerate(inputs):
                result_token, loss = generate_text(
                                model,
                                in_sentence,
                                attention_mask[i][:],
                                eos_id,
                                20,
                                labels[i][:],
                                tokenizer)
                eval_loss += loss.item()
                print("result", result_token)
                print('shape result', result_token.shape)
                print(tokenizer.decode(result_token[0]))
                preds_one.extend(result_token[0].cpu().tolist())
            preds.extend(preds_one.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

            progress_bar.set_postfix(loss=eval_loss / (len(progress_bar)))

    print('***************************', preds[1])
    print('***************************', type(preds[0]))
    preds_text = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    print(preds_text[0])
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
