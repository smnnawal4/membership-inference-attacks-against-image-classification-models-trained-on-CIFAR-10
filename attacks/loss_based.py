import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt


@torch.no_grad()
def per_sample_ce_loss(model, dataset, device="cpu", batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    model.to(device)
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        l = criterion(logits, y)
        losses.append(l.cpu().numpy())

    return np.concatenate(losses)


def sweep_thresholds2(in_losses, out_losses, n_thresh=50):
    losses = np.concatenate([in_losses, out_losses])
    labels = np.concatenate([
        np.ones_like(in_losses),
        np.zeros_like(out_losses)
    ])
    thresholds = np.quantile(
        losses,
        np.linspace(0.01, 0.99, n_thresh)
    )
    precisions = []
    recalls = []
    for thr in thresholds:
        preds = (losses <= thr).astype(int)
        precisions.append(precision_score(labels, preds, zero_division=0))
        recalls.append(recall_score(labels, preds, zero_division=0))

    return thresholds, np.array(precisions), np.array(recalls)



def plot_loss_distributions(in_losses, out_losses, bins=60, max_loss=0.01):
    plt.figure()
    plt.hist(in_losses, bins=bins, range=(0, max_loss),
             density=True, alpha=0.6, label="Members (IN)")
    plt.hist(out_losses, bins=bins, range=(0, max_loss),
             density=True, alpha=0.6, label="Non-members (OUT)")
    plt.xlabel("cross-entropy loss (clipped)")
    plt.ylabel("density")
    plt.title("Loss distribution: IN vs OUT")
    plt.legend()
    plt.show()


def plot_precision_recall_vs_threshold2(thresholds, precisions, recalls):
    plt.figure()
    q = np.linspace(0.01, 0.99, len(thresholds))
    plt.plot(q, precisions, label="Precision")
    plt.plot(q, recalls, label="Recall")
    plt.xlabel("quantile of loss distribution")
    plt.ylabel("score")
    plt.title("Loss-based MIA: Precision / Recall vs threshold")
    plt.legend()
    plt.show()



def plot_roc_and_auc2(in_losses, out_losses):
    scores = -np.concatenate([in_losses, out_losses])  
    labels = np.concatenate([
        np.ones_like(in_losses),
        np.zeros_like(out_losses)
    ])
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (loss-based MIA)")
    plt.legend()
    plt.show()

    return auc
