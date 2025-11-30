from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision

from src.model_training.lost_and_found_dataset import LostAndFoundSegDataset


def get_dataloaders(batch_size: int = 2, val_split: float = 0.1):
    """
    Create train/val dataloaders from the Lost & Found train split.
    """
    full_dataset = LostAndFoundSegDataset(split="train")

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader


def build_model(num_classes: int = 256):
    """
    Build a DeeplabV3 model for semantic segmentation.

    num_classes:
      Lost & Found labelIds are 0–255, so using 256 classes is a simple baseline.
    """
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights="DEFAULT"
    )
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss() 

    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)          
        labels = labels.to(device)         
        optimizer.zero_grad()
        outputs = model(images)["out"]    
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Compute average loss on val set.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)["out"]
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = get_dataloaders(batch_size=2, val_split=0.1)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_model(num_classes=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 2  # start small just to test

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    ckpt_path = Path("checkpoints")
    ckpt_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_path / "deeplabv3_lostandfound_baseline.pth")
    print("Saved checkpoint to", ckpt_path / "deeplabv3_lostandfound_baseline.pth")


if __name__ == "__main__":
    main()