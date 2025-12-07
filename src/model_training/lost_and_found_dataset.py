import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2 

from src.datasets.lost_and_found import list_image_label_pairs, get_laf_id_map

class LostAndFoundInstanceDataset(Dataset):
    """
    Instance Segmentation dataset for Lost & Found (formatted for Mask R-CNN).
    """

    def __init__(self, split: str = "train", transform=None):
        self.split = split
        self.transform = transform
        self.pairs = list_image_label_pairs(split)
        self.id_map = get_laf_id_map()  # Get the map from File 1

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, label_path = self.pairs[idx]

        # 1. Load Image
        img = Image.open(img_path).convert("RGB")
        # Note: In a real pipeline, transforms usually handle to-tensor conversion.
        # Here we do a basic conversion if no transforms are provided.
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # 2. Load Label & Map IDs
        label_raw = np.array(Image.open(label_path), dtype=np.uint8)
        
        # Apply the mapping (Raw ID -> Train ID) using a lookup table
        # 0=Background, 1=Free Space, 2=Hazard
        lut = np.zeros(256, dtype=np.uint8)
        for k, v in self.id_map.items():
            lut[k] = v
        label_mapped = lut[label_raw]

        # 3. Process Instances for Mask R-CNN
        # We only care about TrainID 2 (Hazards) for the detector
        hazard_mask = (label_mapped == 2).astype(np.uint8)

        # Find individual objects (blobs) in the hazard mask
        # This converts a semantic mask into instance masks
        contours, _ = cv2.findContours(hazard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        masks = []
        labels = []

        for contour in contours:
            # Filter tiny noise if necessary (e.g., area < 10 pixels)
            if cv2.contourArea(contour) < 50:
                continue

            # Create binary mask for this specific object
            mask = np.zeros_like(hazard_mask)
            cv2.drawContours(mask, [contour], -1, 1, thickness=-1)

            # Get Bounding Box (x_min, y_min, x_max, y_max)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Mask R-CNN expects: [x0, y0, x1, y1]
            boxes.append([x, y, x+w, y+h])
            masks.append(mask)
            labels.append(1) # Class 1 = "Debris/Hazard" (The model's internal class ID)

        # 4. Handle Empty Images (No hazards found)
        if len(boxes) == 0:
            # PyTorch requires specific format even for empty targets
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, label_mapped.shape[0], label_mapped.shape[1]), dtype=torch.uint8),
                "image_id": torch.tensor([idx])
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
                "image_id": torch.tensor([idx])
            }

        return img_tensor, target

if __name__ == "__main__":
    # Quick Test
    ds = LostAndFoundInstanceDataset(split="train")
    img, target = ds[0]
    print(f"Image shape: {img.shape}")
    print(f"Target keys: {target.keys()}")
    print(f"Number of objects found: {len(target['boxes'])}")