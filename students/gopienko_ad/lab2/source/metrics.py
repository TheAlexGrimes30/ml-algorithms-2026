import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray,) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.mean(y_true == y_pred))

def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true != 1) & (y_pred == 1))

    return float(tp / (tp + fp + eps))

def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred != 1))

    return float(tp / (tp + fn + eps))


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    p = precision(y_true, y_pred, eps=eps)
    r = recall(y_true, y_pred, eps=eps)

    return float(2 * p * r / (p + r + eps))

def roc_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    positive_label=1,
) -> float:

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    positive_mask = (y_true == positive_label)
    negative_mask = (y_true != positive_label)

    n_positive = int(np.sum(positive_mask))
    n_negative = int(np.sum(negative_mask))

    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort",)
    sorted_scores = scores[order]

    ranks = np.zeros(len(scores), dtype=float)

    left = 0

    while left < len(scores):
        right = left + 1

        while right < len(scores) and sorted_scores[right] == sorted_scores[left]:
            right += 1

        average_rank = ((left + 1) + right) / 2.0
        ranks[order[left:right]] = average_rank
        left = right

    positive_rank_sum = np.sum(ranks[positive_mask])
    auc = (positive_rank_sum - n_positive * (n_positive + 1)/ 2.0) / (n_positive * n_negative)
    return float(auc)