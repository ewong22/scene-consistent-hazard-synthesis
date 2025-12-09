import numpy as np
from PIL import Image
from src.data_generation.generator import HazardGenerator

def test_parsing_logic():
    print("--- Testing Parse Logic ---")
    gen = HazardGenerator()
    W, H = 2048, 1024 # Standard Cityscapes/LostAndFound size

    # Test Case 1: Standard Integers (What you expected before)
    input_int = "[100, 100, 200, 200]"
    res_int = gen.parse_box(input_int, W, H)
    print(f"Input: {input_int} -> Output: {res_int}")
    assert res_int == (100, 100, 200, 200), "Failed Integer Test"

    # Test Case 2: Floats (The likely culprit of your bug)
    input_float = "[0.1, 0.1, 0.2, 0.2]" 
    # 0.1 * 2048 = 204, 0.1 * 1024 = 102
    res_float = gen.parse_box(input_float, W, H)
    print(f"Input: {input_float} -> Output: {res_float}")
    # We accept slight rounding differences
    assert res_float[0] > 200 and res_float[1] > 100, "Failed Float Scaling Test"

    # Test Case 3: Noisy Text (VLM chatter)
    input_noise = "Sure! I found a spot at box [50, 50, 100, 100]."
    res_noise = gen.parse_box(input_noise, W, H)
    print(f"Input: {input_noise} -> Output: {res_noise}")
    assert res_noise == (50, 50, 100, 100), "Failed Noisy Text Test"

    print("✅ All Logic Tests Passed!\n")
    return gen

def test_generation_pipeline(gen):
    print("--- Testing Full Generation Pipeline ---")
    # 1. Create Dummy Data (So we don't rely on paths existing)
    print("Creating dummy input images...")
    W, H = 512, 256
    dummy_img = Image.new("RGB", (W, H), color="gray")
    dummy_depth = Image.new("RGB", (W, H), color="black")
    
    dummy_img.save("test_input.png")
    dummy_depth.save("test_depth.png")

    # 2. Run Generation
    # We use integers here to ensure the box is valid for this small image
    vlm_box = "[50, 50, 150, 150]"
    prompt = "A red wooden crate"

    print("Running generate_hazard()...")
    try:
        result = gen.generate_hazard(
            image_path="test_input.png",
            depth_map_path="test_depth.png",
            vlm_box_text=vlm_box,
            prompt=prompt,
            output_path="test_result.png",
            debug=True # This will produce debug_mask.png
        )
        print("Generation finished.")
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        return

    # 3. Verify Output Mask (The most important part)
    if not os.path.exists("debug_mask.png"):
        print("❌ FAILURE: debug_mask.png was not created.")
        return

    mask_arr = np.array(Image.open("debug_mask.png"))
    unique = np.unique(mask_arr)
    print(f"Mask Unique Values: {unique}")
    
    if 255 in unique:
        print("✅ SUCCESS: Mask contains white pixels (255). The box is being drawn!")
    else:
        print("❌ FAILURE: Mask is all black. Parsing likely failed or box size is 0.")

if __name__ == "__main__":
    import os
    gen_instance = test_parsing_logic()
    test_generation_pipeline(gen_instance)