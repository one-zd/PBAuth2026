import os, torch, time, argparse, gc, sys
import torch.nn.functional as F
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from transformers import CLIPImageProcessor
from vine.src.Alias.training_src.transformations import TransformNet
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import torch.cuda.amp as amp

try:
    import lpips
except ImportError:
    lpips = None
    print("Warning: Could not import lpips. LPIPS loss will be disabled if requested.")

try:
    from vine.src.Alias.other.color_l2 import ciede2000_loss
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'other'))
        from color_l2 import ciede2000_loss
    except ImportError:
        print("Warning: Could not import ciede2000_loss. Color loss will be disabled.")
        ciede2000_loss = None

try:
    from vine.src.Alias.private_safety_checker import PrivateDifferentiableSafetyChecker
except ImportError:
    try:
        from private_safety_checker import PrivateDifferentiableSafetyChecker
    except ImportError:
        print("Error: Could not import PrivateDifferentiableSafetyChecker.")
        sys.exit(1)

class ImageDataset(Dataset):
    def __init__(self, source_dir, file_list):
        self.source_dir = source_dir
        self.file_list = file_list
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        f = self.file_list[idx]
        img_path = os.path.join(self.source_dir, f)
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, f
        except Exception as e:
            print(f"Error loading {f}: {e}")
            return torch.zeros(3, 512, 512), f

def save_batch_images(batch_tensors, batch_files, output_func):
    to_pil = transforms.ToPILImage()
    for i, f in enumerate(batch_files):
        try:
            save_path = output_func(f)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            to_pil(batch_tensors[i]).save(save_path)
        except Exception as e:
            print(f"Error saving {f}: {e}")

def process_batch(source_dir, output_func, file_list):
    global total_time
    
    # Filter out files that already exist
    files_to_process = []
    for f in file_list:
        if not os.path.exists(output_func(f)):
            files_to_process.append(f)
        else:
            # print(f"Skipping {f}, already exists.")
            pass
            
    if not files_to_process:
        return

    # Use DataLoader for parallel loading
    dataset = ImageDataset(source_dir, files_to_process)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    # scaler = amp.GradScaler()
    executor = ThreadPoolExecutor(max_workers=8)

    for batch_images, batch_files in tqdm(dataloader, desc="Processing batches"):
        input_tensor = batch_images.to(device, non_blocking=True)
        
        start_time = time.time()
        adv_image_tensor = input_tensor.clone().detach()
        
        for step in range(args.steps):
            adv_image_tensor.requires_grad = True
            
            # with amp.autocast():
            if args.use_robust_attack:
                adv_input_trans = adv_image_tensor * 2.0 - 1.0
                simulated_global_step = int(step * (200 / args.steps))
                attacked_image = transform_net(adv_input_trans, adv_input_trans, global_step=simulated_global_step, args=args)
                attacked_image = (attacked_image + 1.0) / 2.0
            else:
                attacked_image = adv_image_tensor
            
            model_input_raw = F.interpolate(attacked_image, size=(224, 224), mode='bilinear', align_corners=False)
            model_input = (model_input_raw - mean) / std
            
            _, nsfw_logits, _, _ = safety_checker(clip_input=model_input, nsfw_threshold=0.0)
            loss_nsfw = -nsfw_logits.mean()

            # Calculate Color Loss (CIEDE2000)
            loss_color = torch.tensor(0.0, device=device)
            if args.color_loss_weight > 0 and ciede2000_loss is not None:
                # ciede2000_loss expects inputs in [0, 1]
                loss_color = ciede2000_loss(adv_image_tensor, input_tensor).mean()
                
            # Calculate LPIPS Loss
            loss_lpips = torch.tensor(0.0, device=device)
            if args.lpips_loss_weight > 0 and loss_fn_lpips is not None:
                # LPIPS expects inputs in [-1, 1]
                adv_norm = adv_image_tensor * 2.0 - 1.0
                orig_norm = input_tensor * 2.0 - 1.0
                loss_lpips = loss_fn_lpips(adv_norm, orig_norm).mean()

            loss = loss_nsfw + args.color_loss_weight * loss_color + args.lpips_loss_weight * loss_lpips
            
            safety_checker.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                grad = adv_image_tensor.grad
                # In-place update to save memory
                adv_image_tensor.data.add_(grad.sign(), alpha=-args.alpha)
                
                # Projection
                # clamp(min, max) works with tensors in newer PyTorch, but to be safe and efficient:
                adv_image_tensor.data = torch.max(torch.min(adv_image_tensor, input_tensor + args.epsilon), input_tensor - args.epsilon)
                adv_image_tensor.data.clamp_(0.0, 1.0)
            
        end_time = time.time()
        total_time += (end_time - start_time)
        
        # Async save
        batch_images_cpu = adv_image_tensor.detach().cpu()
        executor.submit(save_batch_images, batch_images_cpu, batch_files, output_func)
        
        # gc.collect()
        # torch.cuda.empty_cache()
        
    executor.shutdown(wait=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="E:\phd//4\code\VINE\W-Bench", help='path to the input image')
    parser.add_argument('--output_dir', type=str, default='E:\phd//4\code\VINE\W_bench_en\PGD_ep48_al2_st200', help='the directory to save the output')
    parser.add_argument("--model_path", type=str, default="E:\phd//4\code\VINE\CompVis\stable-diffusion-safety-checker", help="Path to safety checker model")
    
    # PGD Args
    parser.add_argument("--epsilon", type=float, default=32/255.0, help="Perturbation budget (L_inf norm)")
    parser.add_argument("--alpha", type=float, default=2/255.0, help="Step size")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--color_loss_weight", type=float, default=0, help="Weight for CIEDE2000 color loss")
    parser.add_argument("--lpips_loss_weight", type=float, default=0, help="Weight for LPIPS loss (VGG)")
    
    # Robustness Args
    parser.add_argument("--use_robust_attack", type=str2bool, default=True, help="Whether to use robust attack with TransformNet")

    parser.add_argument("--imagenetc_step", type=int, default=80)   #####if step=100，default=80. step>200,default=160
    parser.add_argument("--crop_resize_step", type=int, default=80)
    parser.add_argument("--ig_filter_step", type=int, default=80)

    parser.add_argument("--rnd_noise_ramp", type=int, default=10)
    parser.add_argument("--contrast_ramp", type=int, default=10)
    parser.add_argument("--rnd_sat_ramp", type=int, default=10)
    parser.add_argument("--rnd_bri_ramp", type=int, default=10)
    parser.add_argument("--rnd_hue_ramp", type=int, default=10)
    parser.add_argument("--jpeg_quality_ramp", type=int, default=10)
    parser.add_argument("--rnd_noise", type=float, default=0.02)
    parser.add_argument("--contrast_low", type=float, default=0.5)
    parser.add_argument("--contrast_high", type=float, default=1.5)
    parser.add_argument("--rnd_sat", type=float, default=1.0)
    parser.add_argument("--rnd_bri", type=float, default=0.3)
    parser.add_argument("--rnd_hue", type=float, default=0.1)
    parser.add_argument("--jpeg_quality", type=int, default=10)
    parser.add_argument("--no_motionBlur", action="store_true")
    parser.add_argument("--no_gaussianNoise", action="store_true")
    parser.add_argument("--no_contrast", action="store_true")
    parser.add_argument("--no_bright", action="store_true")
    parser.add_argument("--no_saturation", action="store_true")
    parser.add_argument("--no_jpeg", action="store_true")
    parser.add_argument("--N_blur", type=int, default=31)

    args = parser.parse_args()

    set_seed(42)
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for faster computation on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Safety Checker
    print(f"Loading Safety Checker from {args.model_path}...")
    safety_checker = PrivateDifferentiableSafetyChecker.from_pretrained(args.model_path)
    feature_extractor = CLIPImageProcessor.from_pretrained(args.model_path)
    safety_checker.to(device)
    safety_checker.eval()
    for param in safety_checker.parameters():
        param.requires_grad = False
        
    # Initialize LPIPS model if needed
    loss_fn_lpips = None
    if args.lpips_loss_weight > 0:
        if lpips is not None:
            print("Loading LPIPS (VGG) model...")
            loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
            loss_fn_lpips.eval()
            for param in loss_fn_lpips.parameters():
                param.requires_grad = False
        else:
            print("Warning: LPIPS requested but library not found. Skipping LPIPS loss.")

    # Load TransformNet
    transform_net = TransformNet(
        device=device,
        rnd_bri=args.rnd_bri, rnd_hue=args.rnd_hue, rnd_sat=args.rnd_sat, rnd_noise=args.rnd_noise,
        do_jpeg=not args.no_jpeg, jpeg_quality=args.jpeg_quality,
        ic_up_level_interval=20 
    ).to(device)
    transform_net.step0.fill_(1)

    # CLIP mean/std
    mean = torch.tensor(feature_extractor.image_mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(feature_extractor.image_std).view(1, 3, 1, 1).to(device)

    resolution = '512'
    total_time = 0
    
    category = ['DET_INVERSION_1K', 'INSTRUCT_1K', 'STO_REGENERATION_1K', 'LOCAL_EDITING_5K', 'SVD_1K', 'DISTORTION_1K', ]
    # category = ['LOCAL_EDITING_5K', 'SVD_1K', 'DISTORTION_1K', ]

    for c in category:
        if c == 'LOCAL_EDITING_5K':
            sub_category = ['10-20', '20-30', '30-40', '40-50', '50-60']
            for cs in sub_category:
                print(cs)
                source = os.path.join(args.input_dir, c, cs,'image')
                files = os.listdir(source)
                output_func = lambda f: os.path.join(args.output_dir, resolution, c, cs, os.path.split(f)[-1][:-4]+'_wm.png')
                process_batch(source, output_func, files)
                
                print(len(files))
                avg_time = total_time/len(files) if len(files) > 0 else 0
                print(f'==============================={avg_time}===============================')
        else:
            # Same logic for other categories
            print(c)
            source = os.path.join(args.input_dir, c, 'image')
            files = os.listdir(source)
            output_func = lambda f: os.path.join(args.output_dir, resolution, c, os.path.split(f)[-1][:-4]+'_wm.png')
            process_batch(source, output_func, files)
                        
            print(len(files))
            avg_time = total_time/len(files) if len(files) > 0 else 0
            print(f'==============================={avg_time}===============================')
                
                