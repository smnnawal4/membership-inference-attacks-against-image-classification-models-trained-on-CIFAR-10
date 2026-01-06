import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_autoencoder(model, dataset, epochs=5, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for x, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()


def train_classifier(
    model, dataset, epochs=20, batch_size=128, lr=1e-3, device="cpu", seed=42
):
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)

        acc = 100.0 * correct / total
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f} - Acc: {acc:.2f}%"
        )


def evaluate_classifier(model, dataset, batch_size=128, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy
