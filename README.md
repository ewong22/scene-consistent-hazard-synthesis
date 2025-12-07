# scene-consistent-hazard-synthesis
scene consistent generative insertion for hazard synthesis

## Setup
1. Download Lost & Found (leftImg8bit + gtCoarse) into `data/raw/lostandfound/`
2. Run the dataset loader test:  
   `python -m src.datasets.lost_and_found`
3. Generate depth maps (optional test):  
   `python -m src.depth_estimation.compute_depth --split train --max-images 10`
4. Generate masks
   `python -m src.generative_insertion.generate_masks`
5. Build the synthetic dataset
   `python -m src.generative_insertion.build_synthetic_dataset`