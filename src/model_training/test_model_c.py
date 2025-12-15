import torch
import cv2
import numpy as np
import random
from pathlib import Path
from src.model_training.model import get_model_instance_segmentation

# --- Configuration ---
MODEL_PATH = "model_c_diffusion.pth"  # Your trained Model C
TEST_IMG_DIR = Path("data/raw/lostandfound/leftImg8bit/test") # Or 'train' if you don't have test downloaded
OUTPUT_NAME = "model_c_prediction_result.png"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CONFIDENCE_THRESHOLD = 0.5  # Only show boxes with > 50% confidence

def main():
    print(f"--- Testing Trained Model C (Inference) ---")
    
    # 1. Load the Model Architecture
    # We use 2 classes (Background + Hazard)
    model = get_model_instance_segmentation(num_classes=2)
    
    # 2. Load the Trained Weights
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Could not find {MODEL_PATH}. Did you finish training?")
        return
        
    print(f"Loading weights from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval() # Set to evaluation mode
    
    # 3. Pick a Random Test Image
    # Try finding images in 'test', fallback to 'train' if empty
    all_images = list(TEST_IMG_DIR.rglob("*.png"))
    if not all_images:
        print("Test directory empty/missing. Using 'train' images for demo...")
        TEST_IMG_DIR_FALLBACK = Path("data/raw/lostandfound/leftImg8bit/train")
        all_images = list(TEST_IMG_DIR_FALLBACK.rglob("*.png"))
        
    if not all_images:
        print("ERROR: No images found to test on!")
        return

    # Pick one random image
    img_path = random.choice(all_images)
    print(f"Testing on image: {img_path.name}")
    
    # 4. Preprocess Image
    # OpenCV loads as BGR, PyTorch needs RGB 0-1 Tensor
    orig_img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb / 255.0).permute(2, 0, 1).float().to(DEVICE)
    
    # 5. Run Inference
    print("Running prediction...")
    with torch.no_grad():
        # Model expects a list of tensors
        predictions = model([img_tensor])
        
    # The output is a list of dictionaries (one per image)
    pred = predictions[0]
    
    # 6. Visualize Results
    # We draw on the original BGR image so colors look right in the saved file
    output_img = orig_img_bgr.copy()
    
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    print(f"Found {len(boxes)} detections total.")
    count = 0
    
    for i, box in enumerate(boxes):
        score = scores[i]
        
        if score > CONFIDENCE_THRESHOLD:
            count += 1
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw Rectangle (Green)
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            label_text = f"Hazard: {score:.2f}"
            cv2.putText(output_img, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"  - Box: {x1},{y1},{x2},{y2} | Score: {score:.4f}")

    if count == 0:
        print("No detections above threshold.")
    
    # 7. Save Result
    cv2.imwrite(OUTPUT_NAME, output_img)
    print(f"\nSuccess! Prediction saved to '{OUTPUT_NAME}'")
    print("Open this image to see what Model C detected.")

if __name__ == "__main__":
    main()