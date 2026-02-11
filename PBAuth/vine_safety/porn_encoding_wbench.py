import os, torch, time, argparse, gc
from vine.src.Alias.vine_safety.vine_turbo_input01 import VINE_Turbo
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="E:\phd//4\code\VINE\W-Bench", help='path to the input image')
    parser.add_argument('--output_dir', type=str, default='E:\phd//4\code\VINE\W_bench_en\our_encoded_wbench1205y', help='the directory to save the output')
    parser.add_argument('--pretrained_model_name', type=str, default='F:\data_phd//4\yuzixiao//finetuning_porn1205//checkpoint-20000', help='pretrained_model_name')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    watermark_encoder = VINE_Turbo(args.pretrained_model_name)
    watermark_encoder.to(device)
    
    t_val_256 = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), 
        transforms.ToTensor(),
    ])
    
    t_val_512 = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC), 
    ])

    ### ============= load message =============
    secret_val = np.ones(1, dtype=int)  # 全1数组 表示未授权数据，能触发安全检测器
    secret_val = torch.tensor(secret_val, dtype=torch.float).unsqueeze(0)
    watermark = secret_val.to(device)
        
    resolution = '512'
    total_time = 0
    with torch.no_grad():
        # category = ['DET_INVERSION_1K', 'INSTRUCT_1K', 'STO_REGENERATION_1K', 'LOCAL_EDITING_5K', 'SVD_1K', 'DISTORTION_1K', ]
        category = ['DET_INVERSION_1K']
        for c in category:
            if c == 'LOCAL_EDITING_5K':
                sub_category = ['10-20', '20-30', '30-40', '40-50', '50-60']
                for cs in sub_category:
                    print(cs)
                    source = os.path.join(args.input_dir, c, cs,'image')
                    files = os.listdir(source)
                    for i in tqdm(files, desc="Processing images"):
                        input_path = os.path.join(source, i)
                        save_loc = os.path.join(args.output_dir, resolution, c, cs, os.path.split(i)[-1][:-4]+'_wm.png')
                        if not os.path.exists(save_loc):
                            input_image = Image.open(input_path).convert('RGB')
                            resized_img = t_val_256(input_image) # 256x256
                            resized_img = 2.0 * resized_img - 1.0
                            input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device) # 512x512
                            input_image = 2.0 * input_image - 1.0
                            resized_img = resized_img.unsqueeze(0).to(device)

                            start_time = time.time()
                            encoded_image_256 = watermark_encoder(resized_img, secret=watermark)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            total_time = total_time + elapsed_time
                            # print('\nEncoding time:', end_time - start_time, 's', '\n (Note that please execute multiple times to get the average time)\n')
                            
                            ### ============= resolution scaling to original size =============
                            residual_256 = encoded_image_256 - resized_img # 256x256
                            residual_512 = t_val_512(residual_256) # 512x512 or original size
                            encoded_image = residual_512 + input_image # 512x512 or original size
                            encoded_image = encoded_image * 0.5 + 0.5
                            encoded_image = torch.clamp(encoded_image, min=0.0, max=1.0)
                            
                            output_pil = transforms.ToPILImage()(encoded_image[0])
                            # save the output image
                            os.makedirs(os.path.join(args.output_dir, resolution, c, cs), exist_ok=True)                           
                            output_pil.save(save_loc)
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            print("The image already exists.")
                    
                    print(len(files))
                    avg_time = total_time/len(files)
                    print(f'==============================={avg_time}===============================')
            else:
                print(c)
                source = os.path.join(args.input_dir, c, 'image')
                files = os.listdir(source)
                for i in tqdm(files, desc="Processing images"):
                    input_path = os.path.join(source, i)
                    save_loc = os.path.join(args.output_dir, resolution, c, os.path.split(i)[-1][:-4]+'_wm.png')
                    if not os.path.exists(save_loc):
                        input_image = Image.open(input_path).convert('RGB')
                        
                        resized_img = t_val_256(input_image) # 256x256
                        resized_img = 2.0 * resized_img - 1.0
                        input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device) # 512x512
                        input_image = 2.0 * input_image - 1.0
                        resized_img = resized_img.unsqueeze(0).to(device)

                        start_time = time.time()
                        encoded_image_256 = watermark_encoder(resized_img, secret=watermark)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        total_time = total_time + elapsed_time
                        # print('\nEncoding time:', end_time - start_time, 's', '\n (Note that please execute multiple times to get the average time)\n')
                        
                        ### ============= resolution scaling to original size =============
                        residual_256 = encoded_image_256 - resized_img # 256x256
                        residual_512 = t_val_512(residual_256) # 512x512 or original size
                        encoded_image = residual_512 + input_image # 512x512 or original size
                        encoded_image = encoded_image * 0.5 + 0.5
                        encoded_image = torch.clamp(encoded_image, min=0.0, max=1.0)
                        
                        output_pil = transforms.ToPILImage()(encoded_image[0])
                        # save the output image                        
                        os.makedirs(os.path.join(args.output_dir, resolution, c), exist_ok=True)
                        output_pil.save(save_loc)
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        print("The image already exists.")
                            
                print(len(files))
                avg_time = total_time/len(files)
                print(f'==============================={avg_time}===============================')
                
                