import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.chrf_score import sentence_chrf
from rouge import Rouge
import evaluate


def compute_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, precision, recall, f1


def compute_bleu(preds, labels, tokenizer):
    bleu_scores = []
    for pred, label in zip(preds, labels):
        pred_tokens = tokenizer.convert_ids_to_tokens(pred)
        label_tokens = tokenizer.convert_ids_to_tokens(label)
        bleu_score = sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(bleu_score)
    return sum(bleu_scores) / len(bleu_scores)

def compute_lrouge(preds, labels, tokenizer):
    rouge = evaluate.load('rouge')
    preds_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    labels_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
    results = rouge.compute(predictions=preds_texts, references=labels_texts)
    return results["rougeL"].mid.fmeasure

def compute_cider(preds, labels, tokenizer):
    cider = evaluate.load("cider")
    preds_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    labels_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
    results = cider.compute(predictions=preds_texts, references=labels_texts)
    return results["score"]

def compute_rouge(preds, labels, tokenizer):
    rouge = Rouge()
    preds_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    labels_texts = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
    scores = rouge.get_scores(preds_texts, labels_texts, avg=True)
    return scores

def compute_meteor(preds, labels, tokenizer):
    meteor_scores = []
    for pred, label in zip(preds, labels):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        meteor_scores.append(meteor_score([label_text], pred_text))
    return sum(meteor_scores) / len(meteor_scores)

def compute_chrf(preds, labels, tokenizer):
    chrf_scores = []
    for pred, label in zip(preds, labels):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        label_text = tokenizer.decode(label, skip_special_tokens=True)
        chrf_scores.append(sentence_chrf(reference=label_text, hypothesis=pred_text))
    return sum(chrf_scores) / len(chrf_scores)

def compute_perplexity(loss, num_batches):
    return math.exp(loss / num_batches)
