import os, torch, time, argparse
from vine.src.Alias.vine_safety.vine_turbo_input01 import VINE_Turbo
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
import numpy as np
import psutil


####set PYTHONPATH=E:\phd\4\code\VINE


def crop_to_square(image):
    width, height = image.size

    min_side = min(width, height)
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image
    
def main(args, device):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    ### ============= load model =============
    watermark_encoder = VINE_Turbo(ckpt_path=args.pretrained_model_name)
    #
    # watermark_encoder = VINE_Turbo.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name)
    watermark_encoder.to(device)

    ### ============= load image =============  
    input_image_pil = Image.open(args.input_path).convert('RGB') # 512x512 
    if input_image_pil.size[0] != input_image_pil.size[1]:
        input_image_pil = crop_to_square(input_image_pil)
    
    size = input_image_pil.size
    t_val_256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    t_val_512 = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    ])
    resized_img = t_val_256(input_image_pil) # 256x256
    resized_img = 2.0 * resized_img - 1.0
    input_image = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device) # 512x512
    input_image = 2.0 * input_image - 1.0
    resized_img = resized_img.unsqueeze(0).to(device)

    ### ============= load message =============

    secret_val = np.ones(100, dtype=int)  # 全1数组 表示未授权数据，能触发安全检测器
    secret_val = torch.tensor(secret_val, dtype=torch.float).unsqueeze(0)
    watermark = secret_val.to(device)


    ### ============= watermark encoding =============
    start_time = time.time()
    encoded_image_256 = watermark_encoder(resized_img, secret=watermark)
    end_time = time.time()
    print('\nEncoding time:', end_time - start_time, 's', '\n (Note that please execute multiple times to get the average time)\n')

    ### ============= resolution scaling to original size =============
    residual_256 = encoded_image_256 - resized_img # 256x256
    residual_512 = t_val_512(residual_256) # 512x512 or original size
    encoded_image = residual_512 + input_image # 512x512 or original size
    encoded_image = encoded_image * 0.5 + 0.5
    encoded_image = torch.clamp(encoded_image, min=0.0, max=1.0)

    ### ============= save the output image =============
    output_pil = transforms.ToPILImage()(encoded_image[0])
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    save_loc = os.path.join(args.output_dir, os.path.split(args.input_path)[-1][:-4]+'_wm.png')
    output_pil.save(save_loc)
    print(f'\nWatermarked image saved at: {save_loc}\n')

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Peak VRAM: {peak_vram:.2f} MB")
    
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**2
    print(f"Current RAM: {ram_usage:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='E:\phd//4\code\VINE\example\input//1.png', help='path to the input image')
    parser.add_argument('--output_dir', type=str, default='E:\phd//4\code\VINE\example/watermarked_img', help='the directory to save the output')
    parser.add_argument('--pretrained_model_name', type=str, default='F:\data_phd//4//finetuning_porn1112\checkpoint-20000', help='pretrained_model_name')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    main(args, device)
    