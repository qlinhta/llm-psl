import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.chrf_score import sentence_chrf
from rouge import Rouge
import evaluate


def compute_bleu(preds, labels):
    bleu = evaluate.load('bleu')
    results = bleu.compute(predictions=preds, references=labels)
    return results["bleu"]


def compute_rouge(preds, labels):
    rouge = Rouge()
    scores = rouge.get_scores(preds, labels, avg=True)
    return scores


def compute_chrf(preds, labels):
    chrf_scores = []
    for pred, label in zip(preds, labels):
        chrf_scores.append(sentence_chrf(reference=label, hypothesis=pred))
    return sum(chrf_scores) / len(chrf_scores)


def compute_perplexity(loss, num_batches):
    return math.exp(loss / num_batches)
