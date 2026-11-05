from rep_spaces import BagOfWords, Tokenizer, TF_IDF
from architectures import LogLinear
from dataset import TweetDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import f1_score



def train(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    predictions = []
    targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)

        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(y.cpu().tolist())

    accuracy = sum(
        p == t for p, t in zip(predictions, targets)
    ) / len(targets)

    macro_f1 = f1_score(
        targets, predictions, average="macro"
    )

    return total_loss / len(targets), accuracy, macro_f1



def main():
    # Hyperparameters
    batch_size = 64
    epochs = 50
    learning_rate = 0.001

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load dataset
    dataset = load_dataset("cardiffnlp/tweet_eval", "emoji")

    tokenizer = Tokenizer.from_pretrained('FacebookAI/roberta-base')
    encoder = BagOfWords(tokenizer)

    # Create dataset splits
    train_dataset = TweetDataset(dataset["train"], encoder)
    val_dataset = TweetDataset(dataset["validation"], encoder)
    test_dataset = TweetDataset(dataset["test"], encoder)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
        )

    test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
        )

    # Initialize model
    model = LogLinear(
        input_dim=len(encoder),
        output_dim=20
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Training
    best_val_f1 = 0

    for epoch in range(epochs):
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

    # Final testing using the best validation checkpoint
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    test_loss, test_acc, test_f1 = evaluate(
        model, test_loader, criterion, device
    )

    print("\nFinal test results:")
    print(f"Loss:     {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")

if __name__ == "__main__": 
    main()