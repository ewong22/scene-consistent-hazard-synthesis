import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info  # <--- NEW IMPORT
from PIL import Image
import os

# 1. Configuration
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

class LocalVLM:
    def __init__(self):
        print(f"Loading VLM: {MODEL_ID}...")
        
        # 4-bit Quantization Config (Stable)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load Model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        
        # Load Processor
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
        
        print("VLM Loaded Successfully (Qwen2-VL 4-bit).")

    def analyze_scene(self, image_path):
        image = Image.open(image_path).convert("RGB")
        
        # --- Task 1: Find valid insertion region ---
        prompt_loc = "Detect a clear, empty road surface area where I can place a small object. Return the bounding box."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_loc},
                ],
            }
        ]
        
        # Prepare inputs using the external utility
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # FIXED LINE: Use the imported function, not self.processor.method
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Generate Coordinates
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        loc_result = output_text.split("assistant")[-1].strip()

        # --- Task 2: Generate Hazard Description ---
        prompt_desc = "Describe a single small road debris item (like a tire, box, or rock) that would look realistic here. Keep it very short."
        
        messages_desc = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_desc},
                ],
            }
        ]
        
        text_input_desc = self.processor.apply_chat_template(messages_desc, tokenize=False, add_generation_prompt=True)
        image_inputs_desc, video_inputs_desc = process_vision_info(messages_desc)
        
        inputs_desc = self.processor(
            text=[text_input_desc],
            images=image_inputs_desc,
            videos=video_inputs_desc,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids_desc = self.model.generate(**inputs_desc, max_new_tokens=50)
            
        desc_result = self.processor.batch_decode(generated_ids_desc, skip_special_tokens=True)[0]
        desc_result = desc_result.split("assistant")[-1].strip()

        return loc_result, desc_result

if __name__ == "__main__":
    # Update path
    test_image_path = "data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000000_000060_leftImg8bit.png" 
    
    if not os.path.exists(test_image_path):
        print(f"Error: Could not find image at {test_image_path}")
    else:
        vlm = LocalVLM()
        loc, desc = vlm.analyze_scene(test_image_path)
        
        print("\n" + "="*30)
        print("VLM RESULTS (Qwen2-VL)")
        print("="*30)
        print(f"Suggested Location: {loc}")
        print(f"Suggested Object:   {desc}")
        print("="*30)

# "data/raw/lostandfound/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000000_000060_leftImg8bit.png" 