import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import time

# Project Imports
from src.model_training.lost_and_found_dataset import LostAndFoundInstanceDataset
from src.model_training.synthetic_dataset import SyntheticInstanceDataset
from src.model_training.model import get_model_instance_segmentation
from src.model_training.train import collate_fn, train_one_epoch

def main():
    # 1. Configuration
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"--- Training Model C (Real + Diffusion Synthetic) on {device} ---")

    num_classes = 2  # 0=Background, 1=Hazard
    batch_size = 2
    num_epochs = 10
    learning_rate = 0.005

    # 2. Data Setup
    print("Loading Real Data (LostAndFound)...")
    ds_real = LostAndFoundInstanceDataset(split="train")
    
    print("Loading Synthetic Data (Diffusion Generated)...")
    # This automatically loads from data/synthetic_model_c via your dataset class
    ds_synth = SyntheticInstanceDataset()
    
    # Validation: Ensure we actually found synthetic data
    if len(ds_synth) == 0:
        print("CRITICAL ERROR: No synthetic data found. Did you run main_pipeline.py?")
        return

    # Combine datasets
    print(f"Real Samples: {len(ds_real)}")
    print(f"Synth Samples: {len(ds_synth)}")
    dataset_combined = ConcatDataset([ds_real, ds_synth])
    
    data_loader = DataLoader(
        dataset_combined, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_fn
    )

    # 3. Model Setup
    print("Initializing Mask R-CNN...")
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # 4. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Decay learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 5. Training Loop
    print("Starting Training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_one_epoch(model, optimizer, data_loader, device, epoch)
        lr_scheduler.step()
        
        # Optional: Save checkpoint every epoch
        # torch.save(model.state_dict(), f"checkpoint_model_c_epoch_{epoch}.pth")

    total_time = time.time() - start_time
    print(f"\nTraining Complete in {total_time/60:.2f} minutes.")

    # 6. Save Final Model
    save_path = "model_c_diffusion.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to '{save_path}'")

if __name__ == "__main__":
    main()