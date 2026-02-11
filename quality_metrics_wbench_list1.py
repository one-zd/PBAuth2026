import os
import argparse
import cv2
import numpy as np
import torchvision
import lpips
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from openpyxl import Workbook
import shutil

# ------------------------ Utility Functions ------------------------

def list_image_files(folder):
    """Return sorted list of image file paths in folder. Accept common extensions."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    return files

def compute_psnr_ssim(decoded, original):
    decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    psnr_value = psnr(decoded_rgb, original_rgb)
    ssim_value, _ = ssim(decoded_rgb, original_rgb, full=True, channel_axis=2)
    return psnr_value, ssim_value

def compute_lpips(decoded, original, loss_fn_alex, device):
    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    decoded = torchvision.transforms.ToTensor()(decoded) * 2 - 1
    original = torchvision.transforms.ToTensor()(original) * 2 - 1
    decoded = decoded.to(device)
    original = original.to(device)
    return loss_fn_alex(decoded, original).item()

def compute_avg(*lists):
    return [np.mean(i) if len(i) > 0 else float('nan') for i in lists]

# ------------------------ Main Function ------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_a = args.folder_a
    folder_b = args.folder_b

    images_a = list_image_files(folder_a)
    images_b = list_image_files(folder_b)

    if len(images_a) != len(images_b):
        print(f"[Warning] Number of images differ: {len(images_a)} vs {len(images_b)}. Matching by sorted order.")

    # ensure same number of images
    num_images = min(len(images_a), len(images_b))
    images_a = images_a[:num_images]
    images_b = images_b[:num_images]

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # temporary folders for FID
    tmp_fake_dir = "./tmp_fake"
    tmp_real_dir = "./tmp_real"
    for d in [tmp_fake_dir, tmp_real_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    psnr_list, ssim_list, lpips_list = [], [], []

    for img_a_path, img_b_path in tqdm(zip(images_a, images_b), total=num_images, desc="Processing"):
        img_a = cv2.imread(img_a_path)
        img_b = cv2.imread(img_b_path)
        if img_a is None or img_b is None:
            print(f"[Warning] Failed to read images: {img_a_path}, {img_b_path}")
            continue
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

        p, s = compute_psnr_ssim(img_a, img_b)
        l = compute_lpips(img_a, img_b, loss_fn_alex, device)

        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)

        # save images for FID
        basename = os.path.basename(img_a_path)
        cv2.imwrite(os.path.join(tmp_fake_dir, basename), img_a)
        cv2.imwrite(os.path.join(tmp_real_dir, basename), img_b)

    avg_p, avg_s, avg_l = compute_avg(psnr_list, ssim_list, lpips_list)

    try:
        fid_value = fid_score.calculate_fid_given_paths([tmp_fake_dir, tmp_real_dir],
                                                        batch_size=args.fid_batch,
                                                        device=str(device),
                                                        dims=2048)
    except Exception as e:
        print(f"[Error] FID calculation failed: {e}")
        fid_value = float('nan')

    print(f"\nOverall metrics:\nPSNR={avg_p:.4f} SSIM={avg_s:.4f} LPIPS={avg_l:.4f} FID={fid_value:.4f}")

    # Save to Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Evaluation Results"
    ws.append(["Folder A", "Folder B", "PSNR", "SSIM", "LPIPS", "FID"])
    ws.append([folder_a, folder_b, float(avg_p), float(avg_s), float(avg_l), float(fid_value)])
    wb.save(args.xlsx_path)
    print(f"Excel saved to: {args.xlsx_path}")

    # Clean up temporary dirs
    for d in [tmp_fake_dir, tmp_real_dir]:
        shutil.rmtree(d)
    print("Temporary folders removed.")

# ------------------------ Script Entry ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_a', type=str, default='E:\phd//4\code\VINE\W-Bench\LOCAL_EDITING_5K//10-20\image', help="First folder of images")
    parser.add_argument('--folder_b', type=str, default='E:\phd//4\code\VINE\W_bench_en//diffusionguard//512\LOCAL_EDITING_5K//10-20', help="Second folder of images")
    parser.add_argument('--xlsx_path', type=str, default='evaluation_list_results.xlsx')
    parser.add_argument('--fid_batch', type=int, default=50)
    args = parser.parse_args()
    main(args)
