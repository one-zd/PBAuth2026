import os
import argparse
import cv2
import numpy as np
import torchvision
import lpips
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from openpyxl import Workbook
import shutil


# ------------------------ Utils ------------------------

def list_image_files(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


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

    return loss_fn_alex(decoded.to(device), original.to(device)).item()


def compute_avg(*lists):
    return [np.mean(i) if len(i) > 0 else float("nan") for i in lists]


# ------------------------ Main ------------------------

def main(args):
    folder_a = args.folder_a
    folder_b = args.folder_b
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files_a = list_image_files(folder_a)
    files_b = list_image_files(folder_b)

    # ---- 自动匹配：取两文件夹共同的文件名 ----
    common_files = sorted(list(set(files_a) & set(files_b)))

    if len(common_files) == 0:
        print("\n❌ 两文件夹没有共同的文件名，无法比较。")
        return

    print(f"\n🔍 共同图片数量: {len(common_files)}")
    print(f"📁 Folder A: {folder_a}")
    print(f"📁 Folder B: {folder_b}")

    if len(files_a) != len(files_b):
        print(f"⚠️ 提示：文件数不同 ({len(files_a)} vs {len(files_b)})，仅比较共同文件。")

    loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    # tmp folders for FID
    tmp_fake = "./tmp_fake"
    tmp_real = "./tmp_real"
    for d in [tmp_fake, tmp_real]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    psnr_list, ssim_list, lpips_list = [], [], []

    for name in tqdm(common_files, desc="Comparing"):
        img_a_path = os.path.join(folder_a, name)
        img_b_path = os.path.join(folder_b, name)

        img_a = cv2.imread(img_a_path)
        img_b = cv2.imread(img_b_path)

        if img_a is None or img_b is None:
            print(f"⚠️ 跳过无法读取的文件: {name}")
            continue

        # Resize if shapes differ
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

        p, s = compute_psnr_ssim(img_a, img_b)
        l = compute_lpips(img_a, img_b, loss_fn_alex, device)

        psnr_list.append(p)
        ssim_list.append(s)
        lpips_list.append(l)

        cv2.imwrite(os.path.join(tmp_fake, name), img_a)
        cv2.imwrite(os.path.join(tmp_real, name), img_b)

    # ---- compute average metrics ----
    avg_p, avg_s, avg_l = compute_avg(psnr_list, ssim_list, lpips_list)

    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [tmp_fake, tmp_real],
            batch_size=args.fid_batch,
            device=str(device),
            dims=2048
        )
    except Exception as e:
        print(f"FID 计算失败: {e}")
        fid_value = float("nan")

    # ---- print results ----
    print("\n====== 📊 最终结果 ======")
    print(f"PSNR : {avg_p:.4f}")
    print(f"SSIM : {avg_s:.4f}")
    print(f"LPIPS: {avg_l:.4f}")
    print(f"FID  : {fid_value:.4f}")

    # ---- save Excel ----
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append(["Folder A", "Folder B", "Common Files", "PSNR", "SSIM", "LPIPS", "FID"])
    ws.append([folder_a, folder_b, len(common_files), avg_p, avg_s, avg_l, fid_value])
    wb.save(args.xlsx_path)
    print(f"\n📄 Excel 已保存到: {args.xlsx_path}")

    # cleanup
    shutil.rmtree(tmp_fake)
    shutil.rmtree(tmp_real)
    print("🧹 清理完成")


# ------------------------ Entry ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_a', type=str, default='E:\phd//4\code\VINE\W_bench_en\our_encoded_wbench1115//512\INSTRUCT_1K', help="First folder of images")
    parser.add_argument('--folder_b', type=str, default='E:\phd//4\code\VINE\W_bench_en_edit\our_en_edited_wbench1115\INSTRUCT_1K\INSTRUCT_Pix2Pix\9', help="Second folder of images")
    parser.add_argument('--xlsx_path', type=str, default='evaluation_list_results.xlsx')
    parser.add_argument('--fid_batch', type=int, default=50)
    args = parser.parse_args()

    main(args)
