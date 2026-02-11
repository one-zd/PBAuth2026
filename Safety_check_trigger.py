import os, torch, argparse
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from transformers import CLIPFeatureExtractor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from private_safety_checker import PrivateDifferentiableSafetyChecker


def main(args, device):
    # ============= load stable diffusion safety checker =============
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "E:\phd//4\code\VINE\CompVis\stable-diffusion-safety-checker"
    ).to(device)

    # safety_checker = PrivateDifferentiableSafetyChecker.from_pretrained(
    #     "E:\phd//4\code\VINE\CompVis\stable-diffusion-safety-checker"
    # ).to(device)

    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "E:\phd//4\code\VINE\clip-vit-base-patch32"
    )

    # ============= load image =============
    t_val_256 = transforms.Compose([
        # transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    image = Image.open(args.input_path).convert("RGB")
    image_tensor = t_val_256(image).unsqueeze(0).to(device)

    # PIL 格式给 feature_extractor
    clip_input = feature_extractor(images=image, return_tensors="pt").to(device)

    # ============= 安全性检测 =============
    out_images, has_nsfw_concept = safety_checker(
        images=image_tensor,
        clip_input=clip_input.pixel_values
    )

    # out_images, nsfw_logits, nsfw_probs, has_nsfw_concept = safety_checker(
    #     clip_input=clip_input.pixel_values, images=image_tensor, blacken_images=False
    # )


    # is_safe = not any(has_nsfw_concept)
    # output_bit = 1 if is_safe else 0
    output_bit = 1 if any(has_nsfw_concept) else 0
    # ============= groundtruth vs output 对比 =============
    # groundtruth_bit = int(args.input_bit)   # 输入的0/1就是groundtruth
    # match = groundtruth_bit == output_bit
    # result = "通过" if match else "不通过"

    # print("Groundtruth:", groundtruth_bit)
    print("Predicted :", output_bit)
    print('has_nsfw_concept',has_nsfw_concept)
    # print('nsfw_logits',nsfw_logits)
    # print("对比结果:", result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='E:\phd//4\code\VINE\W_bench_en\PGD_ep32_al2_st200_hsv0.5//512\LOCAL_EDITING_5K//10-20//0_2328_wm.png',
                        help='path to the image')
    parser.add_argument('--input_bit', type=str, default='0',
                        help='Groundtruth (0 或 1(unsafe))')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    main(args, device)
