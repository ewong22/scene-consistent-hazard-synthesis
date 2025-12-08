import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.model_training.model import get_model_instance_segmentation
from src.datasets.lost_and_found import list_image_label_pairs, get_laf_id_map

# 1. Configuration
MODEL_PATH = "model_a_real_only.pth"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CONFIDENCE_THRESHOLD = 0.5  # Only show detections with > 50% confidence

def load_test_image(idx=0):
    """
    Manually loads one raw image from the test set without using the full Dataset class
    so we can keep it simple for visualization.
    """
    pairs = list_image_label_pairs("test")
    if not pairs:
        print("No test images found! Switching to 'train' just for a demo.")
        pairs = list_image_label_pairs("train")
    
    img_path, _ = pairs[idx]
    
    # Load Image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize for PyTorch (0-1 range)
    img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    return img, img_tensor

def show_result(img_np, prediction):
    """
    Draws boxes and masks on the image.
    """
    # Clone the image so we can draw on it
    result_img = img_np.copy()
    
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    scores = prediction[0]['scores'].cpu().detach().numpy()
    masks = prediction[0]['masks'].cpu().detach().numpy()

    # Filter by confidence
    keep_indices = scores > CONFIDENCE_THRESHOLD
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    masks = masks[keep_indices]

    print(f"Found {len(boxes)} hazards with score > {CONFIDENCE_THRESHOLD}")

    for i in range(len(boxes)):
        box = boxes[i].astype(int)
        score = scores[i]
        
        # 1. Draw Box (Red)
        cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        
        # 2. Draw Label
        text = f"Hazard: {score:.2f}"
        cv2.putText(result_img, text, (box[0], box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 3. Draw Mask (Optional: Overlay transparent color)
        # Masks from Mask R-CNN are soft (0-1), threshold them at 0.5
        mask = masks[i, 0] > 0.5
        
        # Create a red overlay
        colored_mask = np.zeros_like(result_img)
        colored_mask[mask] = [255, 0, 0]
        
        # Blend overlay with original image
        result_img = cv2.addWeighted(result_img, 1, colored_mask, 0.5, 0)

    plt.figure(figsize=(12, 8))
    plt.imshow(result_img)
    plt.axis('off')
    plt.title(f"Predictions (Threshold: {CONFIDENCE_THRESHOLD})")
    plt.show()

def main():
    # 2. Load Model
    print(f"Loading weights from {MODEL_PATH}...")
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    
    # Load the state dict (weights)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode

    # 3. Run Inference
    # Change 'idx' to see different images
    img_np, img_tensor = load_test_image(idx=10) 
    
    with torch.no_grad():
        prediction = model([img_tensor.to(DEVICE)])

    # 4. Visualize
    show_result(img_np, prediction)

if __name__ == "__main__":
    main()

# python -m src.model_training.visualize