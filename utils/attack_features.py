import torch
import numpy as np
import torch.nn.functional as F


def extract_attack_features(model, dataset, device, batch_size=128):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.eval()

    features = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            features.append(probs.cpu().numpy())

    return np.concatenate(features)
