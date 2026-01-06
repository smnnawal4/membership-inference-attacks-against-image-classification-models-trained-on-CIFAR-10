import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt


@torch.no_grad()
def max_posterior_scores(model, dataset, device="cpu", batch_size=128):
    """score(x) = max_k softmax(model(x))_k"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    scores = []
    for x, _ in loader:
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)
        scores.append(probs.max(dim=1).values.cpu().numpy())
    return np.concatenate(scores)


def sweep_thresholds(in_scores, out_scores, n_thresholds=50):
    """Compute precision/recall for a range of thresholds"""
    scores = np.concatenate([in_scores, out_scores])
    labels = np.concatenate([np.ones_like(in_scores), np.zeros_like(out_scores)])
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    precisions, recalls = [], []
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        precisions.append(precision_score(labels, preds, zero_division=0))
        recalls.append(recall_score(labels, preds, zero_division=0))

    return thresholds, np.array(precisions), np.array(recalls)


def plot_score_distributions(in_scores, out_scores, bins=50):
    """Histogram members vs non-members (distribution plot)."""
    plt.figure()
    plt.hist(in_scores, bins=bins, alpha=0.6, label="Members (IN)", density=True)
    plt.hist(out_scores, bins=bins, alpha=0.6, label="Non-members (OUT)", density=True)
    plt.xlabel("max posterior")
    plt.ylabel("density")
    plt.legend()
    plt.title("Max posterior distribution: IN vs OUT")
    plt.show()


def plot_precision_recall_vs_threshold(thresholds, precisions, recalls):
    """Precision/Recall curves vs threshold."""
    plt.figure()
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.xlabel("threshold on max posterior")
    plt.ylabel("score")
    plt.legend()
    plt.title("Attack performance vs threshold")
    plt.show()


def plot_roc_and_auc(in_scores, out_scores):
    """ROC curve + AUC"""
    scores = np.concatenate([in_scores, out_scores])
    labels = np.concatenate([np.ones_like(in_scores), np.zeros_like(out_scores)])

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("ROC curve (confidence-based MIA)")
    plt.show()

    return auc
