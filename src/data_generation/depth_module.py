import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import cv2
import os

class DepthEstimator:
    def __init__(self):
        print("Loading Depth Model (Depth Anything Small)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Checkpoint: LiheYoung/depth-anything-small-hf
        model_id = "LiheYoung/depth-anything-small-hf"
        
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, 
            use_safetensors=True
        )
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Depth Model Loaded on {self.device}.")

    def estimate_depth(self, image_path):
        """
        Returns a normalized depth map (numpy array) where 0=far, 255=near.
        """
        image = Image.open(image_path).convert("RGB")
        
        # Prepare input
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy
        output = prediction.squeeze().cpu().numpy()

        # Normalize to 0-255 for visualization/saving
        depth_min = output.min()
        depth_max = output.max()
        
        # Depth Anything outputs "Metric Depth" (higher = farther) or "Inverse Depth" (higher = closer)
        # depending on training. Visual inspection of your output suggests we need standard normalization.
        # We normalize so 255 is NEAR (white) and 0 is FAR (black).
        
        depth_normalized = (output - depth_min) / (depth_max - depth_min)
        
        # Invert if necessary (usually Depth Anything output is already correct for visual depth)
        # But for ControlNet, we generally want White=Near, Black=Far.
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        return depth_uint8

if __name__ == "__main__":
    # Test on the image with the box and ball
    test_image_path = "data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000000_000210_leftImg8bit.png" 
    
    if not os.path.exists(test_image_path):
        print(f"Error: Could not find image at {test_image_path}")
    else:
        estimator = DepthEstimator()
        depth_map = estimator.estimate_depth(test_image_path)
        
        # Save the result
        save_path = "test_depth_map.png"
        cv2.imwrite(save_path, depth_map)
        print(f"Success! Depth map saved to {save_path}")