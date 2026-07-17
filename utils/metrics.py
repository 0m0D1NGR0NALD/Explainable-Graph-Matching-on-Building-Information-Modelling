import torch
import numpy as np
import pygmtools


def compute_metrics(S_pred, P_gt, threshold=0.5):
    """
    Paper Section IV-A: Binary classification over all N2 × N1 candidate node pairs.
    """
    N1, N2 = P_gt.shape

    S_real = S_pred[:, :N2]
    hard_assign = pygmtools.hungarian(S_real)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in range(N1):
        for j in range(N2):
            pred_is_match = (hard_assign[i, j] == 1)
            gt_is_match = (P_gt[i, j] == 1)

            if gt_is_match and pred_is_match:
                tp += 1
            elif gt_is_match and not pred_is_match:
                fn += 1
            elif not gt_is_match and pred_is_match:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def permutation_loss(S_pred, P_gt):
    """
    Permutation Loss - Binary Cross-Entropy applied element-wise.
    """
    N1, N2 = P_gt.shape
    S_real = S_pred[:, :N2]

    loss = -(P_gt * torch.log(S_real + 1e-8) + (1 - P_gt) * torch.log(1 - S_real + 1e-8)).mean()

    return loss