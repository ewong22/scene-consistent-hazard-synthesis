import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

# Import your project modules
from src.model_training.lost_and_found_dataset import LostAndFoundInstanceDataset
from src.model_training.model import get_model_instance_segmentation
from src.model_training.train import collate_fn

# Configuration
MODEL_PATH = "model_a_real_only.pth"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def evaluate_model():
    print(f"--- Starting Evaluation for Model A ---")
    print(f"Device: {DEVICE}")

    # 1. Load Data (Test Split)
    # NOTE: Ensure you have downloaded the 'test' folder in data/raw/lostandfound/leftImg8bit/test
    try:
        dataset_test = LostAndFoundInstanceDataset(split="test")
        print(f"Test Set Size: {len(dataset_test)} images")
    except FileNotFoundError:
        print("ERROR: Test dataset not found. Please download the LostAndFound 'test' split.")
        print("For now, testing on 'train' split just to verify the code works...")
        dataset_test = LostAndFoundInstanceDataset(split="train")

    data_loader = DataLoader(
        dataset_test, 
        batch_size=1, # Eval is typically done 1 image at a time
        shuffle=False, 
        num_workers=2,
        collate_fn=collate_fn
    )

    # 2. Load Model
    num_classes = 2 # Background + Hazard
    model = get_model_instance_segmentation(num_classes)
    
    # Load weights
    if not torch.cuda.is_available():
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(MODEL_PATH)
        
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 3. Initialize Metric Calculator
    # We calculate metrics for both Bounding Boxes (BBox) and Segmentation Masks (Segm)
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    
    # If you want mask metrics too (as per PDF), you can run a second metric or toggle iou_type
    # But let's start with Box AP as the primary indicator.
    
    print("Running Inference...")
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            # Move to device
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Predict
            outputs = model(images)

            # Update Metric
            # torchmetrics expects a list of prediction dicts and target dicts
            metric.update(outputs, targets)

    # 4. Compute and Print Results
    print("\nComputing Metrics (this may take a moment)...")
    results = metric.compute()

    print("\n" + "="*40)
    print("MODEL A EVALUATION RESULTS (Real Data Only)")
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