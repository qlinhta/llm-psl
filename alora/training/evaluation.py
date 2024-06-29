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


def compute_bleu(preds, labels):
    bleu = evaluate.load('bleu')
    results = bleu.compute(predictions=preds, references=labels)
    return results["bleu"]

def compute_lrouge(preds, labels):
   
    results = rouge.compute(predictions=preds, references=labels)
    return results["rougeL"].mid.fmeasure

def compute_cider(preds, labels):
    cider = evaluate.load("cider")
    results = cider.compute(predictions=preds, references=labels)
    return results["score"]

def compute_rouge(preds, labels):
    rouge = Rouge()
    scores = rouge.get_scores(preds, labels, avg=True)
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