import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils 

from src.model_training.lost_and_found_dataset import LostAndFoundInstanceDataset
from src.model_training.model import get_model_instance_segmentation

def collate_fn(batch):
    """
    Custom collate function is REQUIRED for Mask R-CNN.
    Default PyTorch collate_fn tries to stack tensors, but our targets 
    are lists of dictionaries (boxes, masks) which cannot be stacked.
    """
    return tuple(zip(*batch))

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(data_loader):
        # 1. Move data to GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 2. Forward Pass
        # During training, the model calculates the loss automatically
        loss_dict = model(images, targets)
        
        # The model returns a dictionary of losses (classifier, box_reg, mask, etc.)
        losses = sum(loss for loss in loss_dict.values())

        # 3. Backward Pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 4. Logging
        running_loss += losses.item()
        if i % 10 == 0:
            print(f"Epoch: [{epoch}] Iteration: [{i}/{len(data_loader)}] Loss: {losses.item():.4f}")

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch}] Complete. Average Loss: {avg_loss:.4f}")

def main():
    # --- Configuration ---
    # Use GPU if available, otherwise CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on: {device}")

    num_classes = 2  # 0=Background, 1=Hazard
    batch_size = 2   # Reduce to 2 or 1 if you run out of GPU memory
    num_epochs = 10
    learning_rate = 0.005

    # --- 1. Data Setup ---
    print("Loading Data...")
    dataset_train = LostAndFoundInstanceDataset(split="train")
    # For a real run, you'd usually use a subset or separate split for validation
    # dataset_val = LostAndFoundInstanceDataset(split="test") 

    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_fn # Important!
    )

    # --- 2. Model Setup ---
    print("Initializing Model...")
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # --- 3. Optimizer Setup ---
    # SGD is the standard optimizer for Mask R-CNN
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Optional: Learning Rate Scheduler (decays LR every 3 epochs)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- 4. Training Loop ---
    print("Starting Training...")
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch)
        
        # Update Learning Rate
        lr_scheduler.step()
        
        # Save checkpoint every epoch (optional)
        # torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

    # --- 5. Save Final Model ---
    torch.save(model.state_dict(), "model_a_real_only.pth")
    print("Training Complete. Model saved to 'model_a_real_only.pth'")

if __name__ == "__main__":
    main()