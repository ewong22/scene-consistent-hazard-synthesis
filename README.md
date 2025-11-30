# scene-consistent-hazard-synthesis
scene consistent generative insertion for hazard synthesis
ß
## Setup
1. Create a Python virtual environment
2. Install requirements: `pip install -r requirements.txt`
3. Download Lost & Found (leftImg8bit + gtCoarse) into `data/raw/lostandfound/`
4. Run the dataset loader test:  
   `python -m src.datasets.lost_and_found`
5. Generate depth maps (optional test):  
   `python -m src.depth_estimation.compute_depth --split train --max-images 3`