import math
import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.chrf_score import sentence_chrf
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def compute_bleu(preds, labels):
    bleu = evaluate.load('bleu')
    results = bleu.compute(predictions=preds, references=labels)
    return results["bleu"]


def compute_meteor(preds, labels):
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=preds, references=labels)
    return results["meteor"]


def compute_rouge(preds, labels):
    rouge = Rouge()
    scores = rouge.get_scores(preds, labels, avg=True)
    return scores


def compute_rouge_l(preds, labels):
    rouge_l = evaluate.load('rouge')
    results = rouge_l.compute(predictions=preds, references=labels)
    return results['rougeL']


def compute_cider(preds, labels):
    gts = {i: [label] for i, label in enumerate(labels)}
    res = {i: [pred] for i, pred in enumerate(preds)}
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return score


def compute_chrf(preds, labels):
    chrf_scores = [sentence_chrf(reference=label, hypothesis=pred) for pred, label in zip(preds, labels)]
    return sum(chrf_scores) / len(chrf_scores)


def compute_perplexity(loss, num_batches):
    return math.exp(loss / num_batches)
