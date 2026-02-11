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
import shutil
from openpyxl import Workbook

# ------------------------ Utility Functions ------------------------

def image_to_tensor(image, normalize=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    if normalize:
        image /= 255.0
    return torch.from_numpy(image)


def computePsnr(encoded_warped, image_input):
    mse = F.mse_loss(encoded_warped, image_input, reduction='none')
    mse = mse.mean([1, 2, 3])
    psnr_val = 10 * torch.log10(1 / mse)
    return psnr_val.mean().item()


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


def list_image_files(folder):
    """Return sorted list of image file paths in folder. Accept common extensions."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]
    return files

# ------------------------ Main Evaluation Function ------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # folders
    input_root = os.path.join(args.input_dir, '512')
    wbench_root = args.wbench_path

    # Restore all categories (ensure SVD_1K is included)
    categories = ['INSTRUCT_1K', 'DET_INVERSION_1K', 'STO_REGENERATION_1K',
                  'LOCAL_EDITING_5K', 'DISTORTION_1K','SVD_1K']
    # categories = ['INSTRUCT_1K']

    sub_categories = ['10-20', '20-30', '30-40', '40-50', '50-60']

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # Initialize Workbook here
    wb = Workbook()
    ws = wb.active
    ws.title = "Evaluation Results"
    ws.append(["category", "sub_category", "PSNR", "SSIM", "LPIPS", "FID"])

    # Results (per-category rows)
    results = []

    # ---------------------------------------
    #   Loop over categories
    # ---------------------------------------
    for category in categories:
        print(f"\n========== {category} ==========")

        if category == 'LOCAL_EDITING_5K':
            for sc in sub_categories:
                print(f"--- Sub-category: {sc} ---")
                image_folder = os.path.join(input_root, category, sc)
                image_paths = list_image_files(image_folder)
                if len(image_paths) == 0:
                    print(f"[Warning] No images found in {image_folder}, skipping.")
                    continue

                local_psnr, local_ssim, local_lpips = [], [], []

                real_dir_source = os.path.join(wbench_root, category, sc, 'image')

                for img_path in tqdm(image_paths, desc=f"{category}/{sc}"):
                    decoded = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if decoded is None:
                        print(f"[Warning] Failed to read decoded image: {img_path}, skipping.")
                        continue

                    original_path = os.path.join(wbench_root, category, sc, 'image',
                                                 os.path.basename(img_path).replace('_wm', ''))
                    if not os.path.exists(original_path):
                        print(f"[Warning] Original not found: {original_path}, skipping.")
                        continue
                    original = cv2.imread(original_path, cv2.IMREAD_COLOR)
                    if original is None:
                        print(f"[Warning] Failed to read original image: {original_path}, skipping.")
                        continue

                    if decoded.shape != original.shape:
                        original = cv2.resize(original, (decoded.shape[1], decoded.shape[0]))

                    p, s = compute_psnr_ssim(decoded, original)
                    l = compute_lpips(decoded, original, loss_fn_alex, device)

                    local_psnr.append(p)
                    local_ssim.append(s)
                    local_lpips.append(l)

                # compute sub-category FID & averages
                if len(local_psnr) == 0:
                    print(f"[Warning] No valid images processed for {category}/{sc}, skipping metrics.")
                    continue

                try:
                    fid_value = fid_score.calculate_fid_given_paths([image_folder, real_dir_source],
                                                                    batch_size=args.fid_batch,
                                                                    device=str(device),
                                                                    dims=2048)
                except Exception as e:
                    print(f"[Error] FID calculation failed for {category}/{sc}: {e}")
                    fid_value = float('nan')

                avg_p, avg_s, avg_l = compute_avg(local_psnr, local_ssim, local_lpips)
                print(f"PSNR={avg_p:.4f} SSIM={avg_s:.4f} LPIPS={avg_l:.4f} FID={fid_value if not np.isnan(fid_value) else 'nan'}")
            
                row = [category, sc, float(avg_p), float(avg_s), float(avg_l), float(fid_value) if not np.isnan(fid_value) else None]
                results.append(row)
                ws.append(row)
            
        else:
            # ---------- Single category ----------
            image_folder = os.path.join(input_root, category)
            image_paths = list_image_files(image_folder)
            if len(image_paths) == 0:
                print(f"[Warning] No images found in {image_folder}, skipping.")
                continue

            local_psnr, local_ssim, local_lpips = [], [], []

            real_dir_source = os.path.join(wbench_root, category, "image")

            for img_path in tqdm(image_paths, desc=category):
                decoded = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if decoded is None:
                    print(f"[Warning] Failed to read decoded image: {img_path}, skipping.")
                    continue

                original_path = os.path.join(wbench_root, category, "image",
                                             os.path.basename(img_path).replace('_wm', ''))
                if not os.path.exists(original_path):
                    print(f"[Warning] Original not found: {original_path}, skipping.")
                    continue
                original = cv2.imread(original_path, cv2.IMREAD_COLOR)
                if original is None:
                    print(f"[Warning] Failed to read original image: {original_path}, skipping.")
                    continue

                if decoded.shape != original.shape:
                    original = cv2.resize(original, (decoded.shape[1], decoded.shape[0]))

                p, s = compute_psnr_ssim(decoded, original)
                l = compute_lpips(decoded, original, loss_fn_alex, device)

                local_psnr.append(p)
                local_ssim.append(s)
                local_lpips.append(l)

            if len(local_psnr) == 0:
                print(f"[Warning] No valid images processed for {category}, skipping metrics.")
                continue

            try:
                fid_value = fid_score.calculate_fid_given_paths([image_folder, real_dir_source],
                                                                batch_size=args.fid_batch,
                                                                device=str(device),
                                                                dims=2048)
            except Exception as e:
                print(f"[Error] FID calculation failed for {category}: {e}")
                fid_value = float('nan')

            avg_p, avg_s, avg_l = compute_avg(local_psnr, local_ssim, local_lpips)
            print(f"PSNR={avg_p:.4f} SSIM={avg_s:.4f} LPIPS={avg_l:.4f} FID={fid_value if not np.isnan(fid_value) else 'nan'}")
            
            row = [category, "all", float(avg_p), float(avg_s), float(avg_l), float(fid_value) if not np.isnan(fid_value) else None]
            results.append(row)
            ws.append(row)

    # ------------------------ Write XLSX ------------------------
    wb.save(args.xlsx_path)
    print(f"\nExcel saved to: {args.xlsx_path}")

    # ------------------------ Clean Temporary Directories ------------------------
    # All directories are cleaned up immediately after use in the loop above.
    pass


# ------------------------ Script Entry ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='E:\phd//4\code\VINE\W_bench_en\PGD_ep32_al2_st200_noattack_HSV1', help='path to the (encoded) watermarked images')
    parser.add_argument('--wbench_path', type=str, default='E:/phd/4/code/VINE/W-Bench')
    parser.add_argument('--xlsx_path', type=str, default='evaluation_results.xlsx')
    parser.add_argument('--fid_batch', type=int, default=50, help='batch size for FID calculation')
    args = parser.parse_args()
    main(args)
