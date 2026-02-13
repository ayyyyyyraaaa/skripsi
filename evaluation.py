import numpy as np

def precision_at_k(ground_truth, predictions, k):
    preds = predictions[:k]
    return len(set(preds) & set(ground_truth)) / k

def recall_at_k(ground_truth, predictions, k):
    preds = predictions[:k]
    return len(set(preds) & set(ground_truth)) / len(ground_truth)

def mean_average_precision(ground_truth, predictions, k):
    score = 0
    hits = 0
    for i, p in enumerate(predictions[:k]):
        if p in ground_truth:
            hits += 1
            score += hits / (i+1)
    return score / len(ground_truth)
