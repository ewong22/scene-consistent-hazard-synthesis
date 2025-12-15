import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import os

# Project Imports
from src.model_training.lost_and_found_dataset import LostAndFoundInstanceDataset
from src.model_training.model import get_model_instance_segmentation
from src.model_training.train import collate_fn

# Configuration
MODEL_PATH = "model_c_diffusion.pth"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def evaluate_model():
    print(f"--- Starting Evaluation for Model C ---")
    print(f"Loading weights from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found. Did training finish?")
        return

    # 1. Load Test Data
    try:
        # Tries to load the official test set
        dataset_test = LostAndFoundInstanceDataset(split="test")
        print(f"Test Set Size: {len(dataset_test)} images")
    except (FileNotFoundError, ValueError):
        print("WARNING: 'test' split not found. Falling back to 'train' split for verification.")
        dataset_test = LostAndFoundInstanceDataset(split="train")

    data_loader = DataLoader(
        dataset_test, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2,
        collate_fn=collate_fn
    )

    # 2. Load Model
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    
    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 3. Metrics
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            outputs = model(images)
            metric.update(outputs, targets)

    # 4. Compute Results
    print("\nComputing Metrics...")
    results = metric.compute()

    print("\n" + "="*40)
    print("MODEL C EVALUATION RESULTS (Diffusion Synthetic)")
    print("="*40)
    print(f"mAP (AP)    : {results['map'].item():.4f}")
    print(f"mAP_50      : {results['map_50'].item():.4f}")
    print(f"mAP_75      : {results['map_75'].item():.4f}")
    print("-" * 40)
    print(f"AP (Large)  : {results['map_large'].item():.4f}")
    print(f"AP (Medium) : {results['map_medium'].item():.4f}")
    print(f"AP (Small)  : {results['map_small'].item():.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()