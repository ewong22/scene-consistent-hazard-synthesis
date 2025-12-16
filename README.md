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



Shared Tools (Used by Both Models)
 
vlm_module.py: Uses the Qwen2-VL model to look at an image and find coordinates [x1, y1, x2, y2] for a safe, empty spot on the road. Both Model B and Model C
 
Model A
 
model.py: It downloads a pre-built brain (Mask R-CNN) from PyTorch.
 
train.py: It loads only LostAndFoundInstanceDataset (Real Data). Saves model_a_real_only.pth.
 
Model B: The Copy-Paste Method
 
create_object_bank.py: Scans original training data, finds existing hazards (like crates or toys), "cuts" them out digitally, and saves them as transparent PNGs in data/object_bank.
 
pipeline_model_b.py: It loops through images, asks the VLM for a spot, picks a random "sticker" from the object bank, and pastes it onto the road. it generated the massive synthetic_model_b dataset.
 
test_model_b.py: just testing
 
Model C: The Diffusion Method
 
main_pipeline.py: Get Image -> Get Depth -> Get VLM Location -> Run Generator -> Save Result.
 
generator.py: Stable Diffusion + ControlNet (DIFFUSION HAS AN ISSUE)
 
depth_module.py: Depth Anything model to create a 3D depth map of the scene
 
Get Depth(from depth_module) -> Get VLM Location(from vlm_module.py) -> Run Generator(from generator)
 
Its generates the massive synthetic_model_c dataset
 