import os
import argparse
import replicate
from PIL import Image
import io
import time

# Ensure you have your REPLICATE_API_TOKEN set in your environment variables.
# e.g. set REPLICATE_API_TOKEN=r8_...


# set REPLICATE_API_TOKEN=r8_dsRc9xDfza2iFxMjNKNcozW1lqiTQ440k8BVb

def edit_one_image_by_api(image_path, mask_path, prompt, output_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"Output already exists at {output_path}. Skipping.")
        return

    try:
        # Load and Resize Image
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
            
        print(f"[Info] Loading image from {image_path}...")
        image = Image.open(image_path).convert("RGB").resize((512, 512))
        
        # Load and Resize Mask
        if mask_path and os.path.exists(mask_path):
            print(f"[Info] Loading mask from {mask_path}...")
            mask = Image.open(mask_path).convert("RGB").resize((512, 512))
        else:
            print("[Info] No mask provided or found, using default white mask (full image edit).")
            # Default white mask if not found
            mask = Image.new("RGB", image.size, (255, 255, 255))

        # Convert images to BytesIO for API transmission
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        mask_byte_arr = io.BytesIO()
        mask.save(mask_byte_arr, format='PNG')
        mask_byte_arr.seek(0)

        # Prepare Input for API
        input_data = {
            "mask": mask_byte_arr,
            "image": img_byte_arr,
            "prompt": prompt,
            "width": 512,
            "height": 512,
            "prompt_strength": 0.1,
            "high_noise_frac": 0.5,
            "refine": "no_refiner",
            "apply_watermark": False,
            "num_inference_steps": 25
        }

        print(f"[Info] Sending to Replicate API (SDXL) with prompt: '{prompt}'...")
        # Run Replicate API
        output = replicate.run(
            "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            input=input_data
        )

        # Log the raw output object for analysis
        log_path = output_path + ".log.txt"
        with open(log_path, "w", encoding='utf-8') as log_file:
            log_file.write(f"Prompt: {prompt}\n")
            log_file.write(f"Time: {time.ctime()}\n")
            log_file.write(f"Raw Output Type: {type(output)}\n")
            log_file.write(f"Raw Output: {str(output)}\n")
        print(f"[Info] API interaction log saved to: {log_path}")

        # Save Output
        if output:
            # Replicate output is usually a list of file-like objects (iterator)
            item = output[0]  
            with open(output_path, "wb") as file:
                file.write(item.read())
            
            print(f"[Success] Edited image saved to: `{output_path}`")
        else:
            print("[Warning] No output received from Replicate API.")

    except Exception as e:
        error_msg = str(e)
        
        # Log error to file as well
        try:
            log_path = output_path + ".error.log"
            with open(log_path, "w", encoding='utf-8') as log_file:
                log_file.write(f"Time: {time.ctime()}\n")
                log_file.write(f"Error: {error_msg}\n")
        except:
            pass

        if "NSFW" in error_msg or "nsfw" in error_msg:
            print(f"\n{'='*40}\n!!! [NSFW DETECTED] !!!\nPotential NSFW content was detected. The API blocked the output.\n{'='*40}\n")
        else:
            print(f"[Error] Failed to process image: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default='E:\\phd\\4\\code\\VINE\\example\\api\\0_2328_wm.png', help="Path to input image")
    parser.add_argument("--mask_path", type=str, default=None, help="Path to input mask (optional)")
    parser.add_argument("--prompt", type=str, default="Replace the pizza with a stack of pancakes and add syrup dripping down", help="Editing prompt")
    parser.add_argument("--output_path", type=str, default='E:\\phd\\4\\code\\VINE\\example\\api\\0_2328_wm_sdxl_edited.png', help="Path to save output image")
    
    args = parser.parse_args()
    
    edit_one_image_by_api(
        image_path=args.image_path,
        mask_path=args.mask_path,
        prompt=args.prompt,
        output_path=args.output_path
    )
