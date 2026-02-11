import os
import argparse
import pandas as pd
import replicate
from PIL import Image
from tqdm import tqdm
import io
import time

# Ensure you have your REPLICATE_API_TOKEN set in your environment variables.
# e.g., set REPLICATE_API_TOKEN=r8_...

def edit_by_api(inputPath_img, inputPath_msk, inputPath_prmt, outputPath, limit=None):
    # Acquire Data and Process Editing:
    os.makedirs(outputPath, exist_ok=True)
    
    # Read CSV
    try:
        df = pd.read_csv(inputPath_prmt)
        # Assuming column 1 is ID and column 2 is Prompt as per reference code
        ID = df.iloc[:, 1].tolist()
        prompts = df.iloc[:, 2].tolist()

        if limit is not None and limit > 0:
            print(f"[Info] Limiting processing to first {limit} images.")
            ID = ID[:limit]
            prompts = prompts[:limit]
    except Exception as e:
        print(f"Error reading prompts file: {e}")
        return

    # Store results for Excel logging
    results_log = []

    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        img_id = str(ID[idx])
        
        # Construct output filename matching reference: {idx}_{ID[idx]}.png
        filename = f"{str(idx)}_{img_id}.png"
        path = os.path.join(outputPath, filename)

        # Construct input paths
        img_name = f"{str(idx)}_{img_id}_wm.png"
        img_full_path = os.path.join(inputPath_img, img_name)
        
        # Mask path typically doesn't have _wm suffix in reference
        mask_name = f"{str(idx)}_{img_id}.png"

        # Prepare log entry
        log_entry = {
            "Index": idx,
            "ID": img_id,
            "Input Image": img_name,
            "Output Image": filename,
            "Prompt": prompt,
            "Status": "Pending",
            "Details": ""
        }
        
        # Check for existing valid output
        if os.path.exists(path):
            log_entry["Status"] = "Skipped"
            log_entry["Details"] = "Already exists"
            results_log.append(log_entry)
            continue
            
        # Check for existing NSFW log (Skip retrying NSFW)
        nsfw_log_path = os.path.join(outputPath, f"{str(idx)}_{img_id}.nsfw.log")
        if os.path.exists(nsfw_log_path):
            log_entry["Status"] = "Skipped (NSFW)"
            log_entry["Details"] = "Previously blocked by NSFW"
            results_log.append(log_entry)
            continue
        
        try:
            # Load and Resize Image
            if not os.path.exists(img_full_path):
                print(f"Image not found: {img_full_path}")
                log_entry["Status"] = "Error"
                log_entry["Details"] = "Input image not found"
                results_log.append(log_entry)
                continue
                
            image = Image.open(img_full_path).convert("RGB").resize((512, 512))
            
            # Load and Resize Mask
            if inputPath_msk:
                mask_full_path = os.path.join(inputPath_msk, mask_name)
            else:
                mask_full_path = ""
            
            if inputPath_msk and os.path.exists(mask_full_path):
                mask = Image.open(mask_full_path).convert("RGB").resize((512, 512))
            else:
                # Default white mask if not found
                mask = Image.new("RGB", image.size, (255, 255, 255))

            # Convert images to BytesIO for API transmission
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            mask_byte_arr = io.BytesIO()
            mask.save(mask_byte_arr, format='PNG')
            mask_byte_arr.seek(0)

            # Prepare Input for API (SDXL)
            input_data = {
                "mask": mask_byte_arr,
                "image": img_byte_arr,
                "prompt": prompt,
                "width": 512,
                "height": 512,
                'guidance_scale': 7.5,
                "prompt_strength": 0.1,
                "high_noise_frac": 0.5,
                "refine": "no_refiner",
                "apply_watermark": False,
                "num_inference_steps": 15
            }

            # Run Replicate API with retry logic for rate limits
            output = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    output = replicate.run(
                        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                        input=input_data
                    )
                    break # Success
                except Exception as e:
                    if "429" in str(e) or "throttled" in str(e):
                        wait_time = 15 * (attempt + 1)
                        print(f"\n[Warning] Rate limit hit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        if attempt == max_retries - 1:
                            raise e # Re-raise if last attempt
                    else:
                        raise e # Re-raise other errors immediately

            # Save Output
            if output:
                # Take the first image
                item = output[0]
                with open(path, "wb") as file:
                    file.write(item.read())
                
                print(f"\t> Edited image {str(idx)}_{img_id} is saved at: `{path}`")
                log_entry["Status"] = "Success"
                log_entry["Details"] = "Saved"
            else:
                # No output received, save log
                print(f"\t> Warning: No output received for {str(idx)}_{img_id}")
                log_entry["Status"] = "No Output"
                log_entry["Details"] = "API returned no result"
                
                log_name = f"{str(idx)}_{img_id}.log"
                log_path = os.path.join(outputPath, log_name)
                try:
                    with open(log_path, "w", encoding='utf-8') as log_file:
                        log_file.write(f"Time: {time.ctime()}\n")
                        log_file.write(f"Prompt: {prompt}\n")
                        log_file.write("Result: No output received from API.\n")
                except:
                    pass

        except Exception as e:
            error_msg = str(e)
            
            # Save error log with specific suffix based on error type
            if "NSFW" in error_msg or "nsfw" in error_msg:
                log_suffix = ".nsfw.log"
                status_code = "NSFW Blocked"
            elif "429" in error_msg or "throttled" in error_msg:
                log_suffix = ".rate_limit.log"
                status_code = "Rate Limited"
            else:
                log_suffix = ".error.log"
                status_code = "Error"

            log_name = f"{str(idx)}_{img_id}{log_suffix}"
            log_path = os.path.join(outputPath, log_name)
            try:
                with open(log_path, "w", encoding='utf-8') as log_file:
                    log_file.write(f"Time: {time.ctime()}\n")
                    log_file.write(f"Prompt: {prompt}\n")
                    log_file.write(f"Error: {error_msg}\n")
            except:
                pass

            if "NSFW" in error_msg or "nsfw" in error_msg:
                print(f"\n{'='*40}\n!!! [NSFW DETECTED] !!!\nPotential NSFW content was detected for image {str(idx)}_{img_id}. The API blocked the output.\n{'='*40}\n")
                log_entry["Status"] = "NSFW Blocked"
                log_entry["Details"] = "NSFW detected"
            elif "429" in error_msg or "throttled" in error_msg:
                print(f"\t> Rate Limit Exceeded for image {str(idx)}_{img_id}: {e}")
                log_entry["Status"] = "Rate Limited"
                log_entry["Details"] = "API throttling"
            else:
                print(f"\t> Error processing image {str(idx)}_{img_id}: {e}")
                log_entry["Status"] = "Error"
                log_entry["Details"] = str(e)
            
            # Optional: sleep to avoid rate limits if hitting hard
            time.sleep(1)
        
        results_log.append(log_entry)
        
        # Periodic Save (Every 10 images)
        if (idx + 1) % 10 == 0:
            try:
                excel_name = "edit_summary.xlsx"
                excel_path = os.path.join(outputPath, excel_name)
                
                df_temp = pd.DataFrame(results_log)
                if not df_temp.empty:
                    df_temp.to_excel(excel_path, sheet_name="Details", index=False)
            except Exception as e:
                print(f"[Warning] Failed to save intermediate Excel: {e}")

        # Enforce rate limit (safe buffer for 6 req/min = 10s per req)
        time.sleep(10)

    # Save summary excel after loop
    try:
        excel_name = "edit_summary.xlsx"
        excel_path = os.path.join(outputPath, excel_name)
        
        df_log = pd.DataFrame(results_log)
        
        if not df_log.empty:
            # Calculate Statistics
            total_count = len(df_log)
            nsfw_count = len(df_log[df_log["Status"] == "NSFW Blocked"])
            # Calculate other statuses if needed
            nsfw_rate = (nsfw_count / total_count) * 100 if total_count > 0 else 0
            
            summary_data = [
                {"Metric": "Total Processed", "Value": total_count},
                {"Metric": "NSFW Blocked", "Value": nsfw_count},
                {"Metric": "NSFW Rate", "Value": f"{nsfw_rate:.2f}%"},
                {"Metric": "Success", "Value": len(df_log[df_log["Status"] == "Success"])},
                {"Metric": "Rate Limited", "Value": len(df_log[df_log["Status"] == "Rate Limited"])},
                {"Metric": "Error", "Value": len(df_log[df_log["Status"] == "Error"])}
            ]
            df_summary = pd.DataFrame(summary_data)

            # Save to multiple sheets
            with pd.ExcelWriter(excel_path) as writer:
                df_log.to_excel(writer, sheet_name="Details", index=False)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)
                
            print(f"\n[Info] Processing summary with NSFW stats saved to {excel_path}")
            print(f"[Stats] NSFW Rate: {nsfw_rate:.2f}% ({nsfw_count}/{total_count})")
        else:
            print("\n[Info] No logs to save.")
            
    except Exception as e:
        print(f"\n[Warning] Failed to save Excel summary: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Default paths matching the reference Global Editing logic (Flat folder structure)
    parser.add_argument("--wm_images_folder", type=str, default=r'E:\phd\4\code\VINE\W_bench_en\PGD_ep32_al2_st200_hsv1\512\INSTRUCT_1K')
    parser.add_argument("--editing_prompt_path", type=str, default=r'E:\phd\4\code\VINE\W-Bench\INSTRUCT_1K\prompts.csv')
    parser.add_argument("--edited_output_folder", type=str, default=r'E:\phd\4\code\VINE\W_bench_en_edit\API_replicate_SDXL\INSTRUCT_1K')
    parser.add_argument("--mask_folder", type=str, default=None, help="Path to mask folder (optional). If not provided, full image edit is performed.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process (default: all(None))")
    args = parser.parse_args()
    
    MODE = "API_SDXL"
    SPEC = "" 

    OUTPUT_PATH = os.path.join(args.edited_output_folder, f"{MODE}{SPEC}")  
        
    print(f"\n>> Processing edited images for [{MODE}{SPEC}]...")
        
    # Check if Prompt file exists before calling
    if not os.path.exists(args.editing_prompt_path):
        print(f"Prompt file not found: {args.editing_prompt_path}. Skipping...")
    else:
        edit_by_api(
            inputPath_img=args.wm_images_folder,
            inputPath_msk=args.mask_folder, # Pass mask folder if available
            inputPath_prmt=args.editing_prompt_path,
            outputPath=OUTPUT_PATH,
            limit=args.limit
        )
