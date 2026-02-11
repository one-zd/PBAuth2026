import os, torch, argparse
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
import numpy as np
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker



def main(args, device):
    ### ============= load model =============
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


    total_samples = 0
    nsfw_triggered = 0
    non_nsfw_files = []
    for root, _, files in os.walk(args.input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            file_name = os.path.basename(input_path)

            image = Image.open(input_path).convert("RGB")
            image_tensor = t_val_256(image).unsqueeze(0).to(device)

            clip_input = feature_extractor(images=image, return_tensors="pt").to(device)
            # ============= 安全性检测 =============
            _, has_nsfw_concept = safety_checker(
                images=image_tensor,
                clip_input=clip_input.pixel_values
            )

            ### ============= watermark decoding & detection =============
            # print(has_nsfw_concept)
            is_nsfw = any(has_nsfw_concept)
            output_bit = 1 if is_nsfw else 0

            # 更新计数器
            total_samples += 1
            if is_nsfw:
                nsfw_triggered += 1
            else:
                non_nsfw_files.append(file_name)  # 记录未触发的文件名

            # 实时计算并打印触发率（保留4位小数）
            # if total_samples > 0:
            #     nsfw_rate = nsfw_triggered / total_samples
            #     print(f"当前样本: {input_path} | NSFW触发率: {nsfw_rate:.4f} ({nsfw_triggered}/{total_samples})")

    # 输出未触发NSFW的图像名称
    print("\n===== 未触发NSFW的图像名称 =====")
    if non_nsfw_files:
        for name in non_nsfw_files:
            print(name)
    else:
        print("所有图像均触发了NSFW")

    final_rate = nsfw_triggered / total_samples
    print(f"\n===== 最终统计 =====")
    print(f"总样本数: {total_samples}")
    print(f"NSFW触发数: {nsfw_triggered}")
    print(f"NSFW触发率: {final_rate:.4f} ({final_rate * 100:.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='E:\phd//4\code\VINE\W_bench_en\PGD_ep32_al2_st200_hsv1_lpips1', help='path to the (edited) watermarked image')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    main(args, device)
    