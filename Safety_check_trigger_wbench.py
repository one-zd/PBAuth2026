import os, torch, argparse
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
import numpy as np
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import pandas as pd
from tqdm import tqdm



def main(args, device):
    ### ============= load model =============
    print("Loading models...")
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "E:\phd//4\code\VINE\CompVis\stable-diffusion-safety-checker"
    ).to(device)

    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "E:\phd//4\code\VINE\clip-vit-base-patch32"
    )

    ### ============= load image =============
    t_val_256 = transforms.Compose([
        # transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    results_list = []

    # Walk through directories
    for root, dirs, files in os.walk(args.input_folder):
        # Filter images
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
        if not image_files:
            continue

        print(f"\nProcessing folder: {root}")
        
        total_samples = 0
        nsfw_triggered = 0
        
        for file in tqdm(image_files, desc="Detecting"):
            input_path = os.path.join(root, file)
            
            try:
                image = Image.open(input_path).convert("RGB")
                image_tensor = t_val_256(image).unsqueeze(0).to(device)

                clip_input = feature_extractor(images=image, return_tensors="pt").to(device)
                
                # ============= 安全性检测 =============
                _, has_nsfw_concept = safety_checker(
                    images=image_tensor,
                    clip_input=clip_input.pixel_values
                )

                is_nsfw = any(has_nsfw_concept)
                
                total_samples += 1
                if is_nsfw:
                    nsfw_triggered += 1
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                continue

        if total_samples > 0:
            accuracy = nsfw_triggered / total_samples
            print(f"Folder: {root} | Accuracy (NSFW Rate): {accuracy:.4f} ({nsfw_triggered}/{total_samples})")
            
            results_list.append({
                "Folder Path": root,
                "Accuracy": accuracy,
                "NSFW Triggered": nsfw_triggered,
                "Total Samples": total_samples
            })

    # Save results to Excel
    if results_list:
        df = pd.DataFrame(results_list)
        output_path = args.output_file
        if not output_path:
             output_path = os.path.join(args.input_folder, "folder_accuracy_report.xlsx")
        
        try:
            df.to_excel(output_path, index=False)
            print(f"\nResults successfully saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving Excel file: {e}")
            # Fallback to CSV if Excel fails (e.g. missing openpyxl)
            csv_path = output_path.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved as CSV instead: {csv_path}")
    else:
        print("\nNo images found or processed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='E:\phd//4\code\VINE\W_bench_en_edit\PGD_ep32_al2_st200_noattack_HSV1', help='path to the (edited) watermarked image')
    parser.add_argument('--output_file', type=str, default=None, help='path to save the xlsx report')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    main(args, device)
    