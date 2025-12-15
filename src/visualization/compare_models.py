import torch
import cv2
import numpy as np
from pathlib import Path
import random

from src.model_training.model import get_model_instance_segmentation
from src.datasets.lost_and_found import get_laf_id_map

# Configuration
MODEL_A_PATH = "model_a_real_only.pth"
MODEL_C_PATH = "model_c_diffusion.pth"
TEST_IMG_DIR = Path("data/raw/lostandfound/leftImg8bit/test")
OUTPUT_DIR = Path("data/visualization_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_model(path):
    model = get_model_instance_segmentation(num_classes=2)
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def visualize_prediction(image_path, model, title):
    # Load Image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb / 255.0).permute(2, 0, 1).float().to(DEVICE)
    
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
    # Draw Boxes (Threshold 0.5)
    output_img = img.copy()
    for i, box in enumerate(prediction['boxes']):
        score = prediction['scores'][i].item()
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_img, f"{score:.2f}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Add Title
    cv2.putText(output_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return output_img

def main():
    print("Loading Models...")
    model_a = load_model(MODEL_A_PATH)
    model_c = load_model(MODEL_C_PATH)
    
    # Pick random test images
    test_images = list(TEST_IMG_DIR.rglob("*.png"))
    if not test_images:
        # Fallback to train if test not downloaded
        test_images = list(Path("data/raw/lostandfound/leftImg8bit/train").rglob("*.png"))
    
    samples = random.sample(test_images, 5)
    
    print(f"Visualizing {len(samples)} comparisons...")
    for i, img_path in enumerate(samples):
        vis_a = visualize_prediction(img_path, model_a, "Model A (Real)")
        vis_c = visualize_prediction(img_path, model_c, "Model C (Synth)")
        
        # Stack vertically
        combined = np.vstack([vis_a, vis_c])
        
        save_path = OUTPUT_DIR / f"comparison_{i}.png"
        cv2.imwrite(str(save_path), combined)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()