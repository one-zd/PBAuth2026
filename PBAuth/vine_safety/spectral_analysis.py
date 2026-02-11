import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse
import os
import sys
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Try to import resnet_our
try:
    import resnet_our
except ImportError:
    print("Warning: resnet_our not found. Please ensure it is in your PYTHONPATH.")
    resnet_our = None

try:
    from vine.src.Alias.private_safety_checker import PrivateDifferentiableSafetyChecker
except ImportError:
    try:
        from private_safety_checker import PrivateDifferentiableSafetyChecker
    except ImportError:
        print("Error: Could not import PrivateDifferentiableSafetyChecker.")

from transformers import CLIPImageProcessor

def get_spectrum_analysis(img_tensor):
    """
    Compute the radial average and 2D magnitude spectrum of an image tensor.
    img_tensor: (C, H, W), values typically in [0, 1] or normalized.
    """
    img_np = img_tensor.detach().cpu().numpy()
    
    # Convert to grayscale for spectrum analysis
    if img_np.shape[0] == 3:
        img_gray = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
    else:
        img_gray = img_np[0]

    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # Radial Profile
    center = (img_gray.shape[0] // 2, img_gray.shape[1] // 2)
    y, x = np.indices((img_gray.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)

    # Average magnitude per radius
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / (nr + 1e-8)
    
    return radialprofile, magnitude_spectrum

def plot_3d_spectrum(magnitude_spectrum, title, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    rows, cols = magnitude_spectrum.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Plot surface
    surf = ax.plot_surface(X, Y, magnitude_spectrum, cmap='viridis', linewidth=0, antialiased=False)
    
    ax.set_title(title)
    ax.set_zlabel('Log Magnitude')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig(save_path)
    plt.close()

def plot_heatmap(magnitude_spectrum, title, save_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude_spectrum, cmap='viridis')
    plt.title(title)
    plt.colorbar(label='Log Magnitude')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def load_resnet(checkpoint_path, device):
    if resnet_our is None:
        raise ImportError("resnet_our module is missing")
    
    print(f"Loading ResNet from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    classes = checkpoint.get('classes', [])
    num_classes = len(classes) if classes else 2 
    
    model = resnet_our.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    state_dict = checkpoint['model']
    # Handle DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()
    return model, classes

def load_safety_checker(model_path, device):
    print(f"Loading Safety Checker from {model_path}...")
    safety_checker = PrivateDifferentiableSafetyChecker.from_pretrained(model_path)
    safety_checker.to(device)
    safety_checker.eval()
    return safety_checker

def pgd_attack_resnet(model, image, target_class_idx, epsilon, alpha, steps, device):
    """
    Targeted PGD attack on ResNet.
    image: (1, C, H, W) in [0, 1]
    """
    # ResNet Normalization
    normalize = transforms.Normalize(mean=[0.5104833, 0.45094156, 0.35637376],
                                     std=[0.17324965, 0.15483698, 0.14012936])
    
    adv_image = image.clone().detach()
    adv_image.requires_grad = True
    
    optimizer = torch.optim.SGD([adv_image], lr=alpha)
    
    print("Running PGD on ResNet...")
    for _ in tqdm(range(steps), desc="PGD ResNet"):
        # Normalize before passing to model
        # Apply normalization to each image in batch
        norm_adv = torch.stack([normalize(t) for t in adv_image])
        
        outputs, _ = model(norm_adv)
        
        # Targeted attack: Minimize CrossEntropy(output, target)
        loss = F.cross_entropy(outputs, torch.tensor([target_class_idx], device=device))
        
        optimizer.zero_grad()
        loss.backward()
        
        data_grad = adv_image.grad.data
        # Gradient Descent for targeted attack (minimize loss)
        adv_image.data = adv_image.data - alpha * data_grad.sign()
        
        # Projection
        eta = torch.clamp(adv_image.data - image.data, -epsilon, epsilon)
        adv_image.data = torch.clamp(image.data + eta, 0, 1)
        
        adv_image.grad.data.zero_()
        
    return adv_image

def pgd_attack_safety_checker(model, processor, image, target_safe, epsilon, alpha, steps, device):
    """
    PGD attack on Safety Checker.
    target_safe: True to make it safe, False to make it unsafe.
    image: (1, C, H, W) in [0, 1]
    """
    adv_image = image.clone().detach()
    adv_image.requires_grad = True
    
    # CLIP Normalization values
    mean = torch.tensor(processor.image_mean).to(device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std).to(device).view(1, 3, 1, 1)
    
    print("Running PGD on Safety Checker...")
    for _ in tqdm(range(steps), desc="PGD Safety Checker"):
        # Normalize
        norm_adv = (adv_image - mean) / std
        
        # Forward
        # nsfw_logits: higher -> more NSFW
        _, nsfw_logits, _, _ = model(norm_adv)
        
        # Loss
        # If we want Safe (NSFW=0), we want to minimize nsfw_logits
        # If we want Unsafe (NSFW=1), we want to maximize nsfw_logits
        
        if target_safe:
            loss = nsfw_logits.sum() # Minimize logits
        else:
            loss = -nsfw_logits.sum() # Maximize logits
            
        loss.backward()
        
        data_grad = adv_image.grad.data
        # Gradient Descent to minimize loss
        adv_image.data = adv_image.data - alpha * data_grad.sign()
        
        # Projection
        eta = torch.clamp(adv_image.data - image.data, -epsilon, epsilon)
        adv_image.data = torch.clamp(image.data + eta, 0, 1)
        
        adv_image.grad.data.zero_()
        
    return adv_image

def main(args):
    device = torch.device(args.device)
    
    # 1. Load Models
    resnet_model, resnet_classes = load_resnet(args.resnet_checkpoint, device)
    safety_model = load_safety_checker(args.safety_model_path, device)
    clip_processor = CLIPImageProcessor.from_pretrained(args.safety_model_path)
    
    # 2. Setup Data
    use_real_images = False
    image_paths = []
    
    if args.input_path and (os.path.isdir(args.input_path) or os.path.isfile(args.input_path)):
        print(f"Loading images from {args.input_path}...")
        if os.path.isfile(args.input_path):
            image_paths.append(args.input_path)
        elif os.path.isdir(args.input_path):
            for root, _, files in os.walk(args.input_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        image_paths.append(os.path.join(root, file))
        
        if len(image_paths) > args.num_samples:
             print(f"Limiting to {args.num_samples} samples.")
             image_paths = image_paths[:args.num_samples]
        
        print(f"Found {len(image_paths)} real images.")
        use_real_images = True
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    else:
        print(f"Generating {args.num_samples} random noise images for analysis...")

    # Determine target class for ResNet
    target_keywords = ['porn', 'nude']
    target_class_indices = []
    for idx, class_name in enumerate(resnet_classes):
        if any(keyword in class_name.lower() for keyword in target_keywords):
            target_class_indices.append(idx)
    
    if not target_class_indices:
        print("Warning: Could not find 'porn' or 'nude' class in ResNet. Defaulting to index 0.")
        target_class_resnet = 0
    else:
        target_class_resnet = target_class_indices[0]
        print(f"Targeting ResNet class: {resnet_classes[target_class_resnet]} (Index: {target_class_resnet})")

    # Accumulators for spectra
    avg_spec_resnet = None
    avg_spec_safety = None
    avg_spec2d_resnet = None
    avg_spec2d_safety = None
    count = 0
    resnet_success_count = 0
    safety_success_count = 0
    
    # Store last perturbations for visualization
    last_img_tensor = None
    last_adv_resnet = None
    last_adv_safety = None
    last_pert_resnet = None
    last_pert_safety = None

    iters = len(image_paths) if use_real_images else args.num_samples
    desc = "Processing Real Images" if use_real_images else "Processing Random Inputs"

    for i in tqdm(range(iters), desc=desc):
        try:
            if use_real_images:
                try:
                    img = Image.open(image_paths[i]).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                except Exception as e:
                    print(f"Error loading {image_paths[i]}: {e}")
                    continue
            else:
                # Generate random noise image [0, 1]
                img_tensor = torch.rand(1, 3, 224, 224).to(device)

            # 3. Generate Perturbations
            
            # ResNet Attack: Target 'porn'
            adv_resnet = pgd_attack_resnet(resnet_model, img_tensor, target_class_resnet, 
                                        args.epsilon, args.alpha, args.steps, device)
            perturbation_resnet = adv_resnet - img_tensor
            
            # Verify ResNet attack success
            with torch.no_grad():
                # Normalize for ResNet
                norm_adv_resnet = torch.stack([transforms.Normalize(mean=[0.5104833, 0.45094156, 0.35637376],
                                     std=[0.17324965, 0.15483698, 0.14012936])(t) for t in adv_resnet])
                res_output, _ = resnet_model(norm_adv_resnet)
                pred_res_idx = res_output.argmax(dim=1).item()
                is_resnet_porn = pred_res_idx == target_class_resnet
            
            # Safety Checker Attack: Target Unsafe (NSFW) -> target_safe=False
            target_safe = False
            adv_safety = pgd_attack_safety_checker(safety_model, clip_processor, img_tensor, 
                                                target_safe, args.epsilon, args.alpha, args.steps, device)
            perturbation_safety = adv_safety - img_tensor
            
            # Verify Safety Checker attack success
            with torch.no_grad():
                 # Mean/Std for CLIP
                mean = torch.tensor(clip_processor.image_mean).to(device).view(1, 3, 1, 1)
                std = torch.tensor(clip_processor.image_std).to(device).view(1, 3, 1, 1)
                norm_adv_safe = (adv_safety - mean) / std
                _, nsfw_logits, nsfw_probs, _ = safety_model(norm_adv_safe)
                # Assuming positive logit or prob > 0.5 means NSFW (Unsafe)
                is_safety_unsafe = nsfw_probs.item() > 0.5

            if not is_resnet_porn:
                print(f"Sample {i}: ResNet attack failed. Pred: {resnet_classes[pred_res_idx]}")
            
            if not is_safety_unsafe:
                print(f"Sample {i}: Safety Checker attack failed. Probability: {nsfw_probs.item():.4f}")

            # Only accumulate if attacks were successful (optional, or just log frequency)
            # User asked to "Ensure", so maybe we should filter? 
            # Or just count success rate. Let's count success rate for report.
            
            if is_resnet_porn:
                resnet_success_count += 1
            if is_safety_unsafe:
                safety_success_count += 1

            # 4. Spectral Analysis
            if is_resnet_porn and is_safety_unsafe:
                spec_resnet, spec2d_resnet = get_spectrum_analysis(perturbation_resnet.squeeze())
                spec_safety, spec2d_safety = get_spectrum_analysis(perturbation_safety.squeeze())
                
                if avg_spec_resnet is None:
                    avg_spec_resnet = spec_resnet
                    avg_spec_safety = spec_safety
                    avg_spec2d_resnet = spec2d_resnet
                    avg_spec2d_safety = spec2d_safety
                else:
                    # Ensure shapes match (should be consistent due to Resize)
                    if len(spec_resnet) == len(avg_spec_resnet):
                        avg_spec_resnet += spec_resnet
                        avg_spec_safety += spec_safety
                        avg_spec2d_resnet += spec2d_resnet
                        avg_spec2d_safety += spec2d_safety
                    else:
                        print(f"Warning: Spectrum size mismatch for sample {i}. Skipping accumulation.")
                        continue
                
                count += 1
                
                # Save last for visualization
                last_img_tensor = img_tensor
                last_adv_resnet = adv_resnet
                last_adv_safety = adv_safety
                last_pert_resnet = perturbation_resnet
                last_pert_safety = perturbation_safety
            else:
                 # If attack failed, we might not want to include it in the average spectrum 
                 # as it doesn't represent a successful trigger.
                 pass

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    if count == 0:
        print("No samples successfully triggered both models.")
        print(f"ResNet Success Rate: {resnet_success_count}/{iters}")
        print(f"Safety Checker Success Rate: {safety_success_count}/{iters}")
        return

    print(f"ResNet Success Rate: {resnet_success_count}/{iters}")
    print(f"Safety Checker Success Rate: {safety_success_count}/{iters}")
    print(f"Used {count} concurrent successful samples for spectral average.")

    # Average
    avg_spec_resnet /= count
    avg_spec_safety /= count
    avg_spec2d_resnet /= count
    avg_spec2d_safety /= count
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 5. Plot 3D Spectrum
    plot_3d_spectrum(avg_spec2d_resnet, '3D Spectrum - ResNet (Target: Porn)', 
                     os.path.join(args.output_dir, '3d_spectrum_resnet.pdf'))
    plot_3d_spectrum(avg_spec2d_safety, '3D Spectrum - Safety (Target: Unsafe)', 
                     os.path.join(args.output_dir, '3d_spectrum_safety.pdf'))

    # 6. Plot Heatmaps
    plot_heatmap(avg_spec2d_resnet, 'Spectrum Heatmap - ResNet', 
                 os.path.join(args.output_dir, 'heatmap_resnet.pdf'))
    plot_heatmap(avg_spec2d_safety, 'Spectrum Heatmap - Safety', 
                 os.path.join(args.output_dir, 'heatmap_safety.pdf'))

    # 7. Plot Difference Heatmap
    # Normalize for difference to be meaningful
    norm_2d_resnet = (avg_spec2d_resnet - avg_spec2d_resnet.min()) / (avg_spec2d_resnet.max() - avg_spec2d_resnet.min() + 1e-8)
    norm_2d_safety = (avg_spec2d_safety - avg_spec2d_safety.min()) / (avg_spec2d_safety.max() - avg_spec2d_safety.min() + 1e-8)
    diff_spec = norm_2d_safety - norm_2d_resnet
    
    plt.figure(figsize=(8, 8))
    plt.imshow(diff_spec, cmap='RdBu_r', vmin=-1, vmax=1) # Red means Safety Higher, Blue means ResNet Higher
    plt.title('Difference Spectrum (Safety - ResNet)')
    plt.colorbar(label='Normalized Log Magnitude Diff')
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'heatmap_difference.pdf'))
    plt.close()

    # 8. Plot Average Radial Spectrum
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Raw Log Magnitude
    plt.subplot(1, 2, 1)
    plt.plot(avg_spec_resnet, label='ResNet (Target: Porn)', linewidth=2)
    plt.plot(avg_spec_safety, label='CLIP Safety (Target: Unsafe)', linewidth=2)
    plt.xlabel('Frequency (Radius)')
    plt.ylabel('Log Magnitude')
    plt.title('Average Radial Power Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Normalized Magnitude2
    # Normalize to [0, 1] range for shape comparison
    norm_spec_resnet = (avg_spec_resnet - avg_spec_resnet.min()) / (avg_spec_resnet.max() - avg_spec_resnet.min() + 1e-8)
    norm_spec_safety = (avg_spec_safety - avg_spec_safety.min()) / (avg_spec_safety.max() - avg_spec_safety.min() + 1e-8)
    
    plt.subplot(1, 2, 2)
    plt.plot(norm_spec_resnet, label='ResNet', linewidth=2)
    plt.plot(norm_spec_safety, label='CLIP Safety', linewidth=2)
    plt.xlabel('Frequency (Radius)')
    plt.ylabel('Normalized Log Magnitude')
    plt.title('Normalized Radial Spectrum Shape')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, 'average_spectral_analysis.pdf')
    plt.savefig(save_path)
    print(f"Analysis plots saved to {args.output_dir}")

    # Save example images (from the last processed image)
    if last_img_tensor is not None:
        to_pil = transforms.ToPILImage()
        
        # Save original
        to_pil(last_img_tensor.squeeze().cpu()).save(os.path.join(args.output_dir, 'example_original.png'))
        
        # Save ResNet Adv & Perturbation
        to_pil(last_adv_resnet.squeeze().cpu()).save(os.path.join(args.output_dir, 'example_adv_resnet.png'))
        pert_res_vis = last_pert_resnet.squeeze().cpu()
        pert_res_vis = (pert_res_vis - pert_res_vis.min()) / (pert_res_vis.max() - pert_res_vis.min() + 1e-8)
        to_pil(pert_res_vis).save(os.path.join(args.output_dir, 'example_perturbation_resnet_norm.png'))

        # Save Safety Adv & Perturbation
        to_pil(last_adv_safety.squeeze().cpu()).save(os.path.join(args.output_dir, 'example_adv_safety.png'))
        pert_safe_vis = last_pert_safety.squeeze().cpu()
        pert_safe_vis = (pert_safe_vis - pert_safe_vis.min()) / (pert_safe_vis.max() - pert_safe_vis.min() + 1e-8)
        to_pil(pert_safe_vis).save(os.path.join(args.output_dir, 'example_perturbation_safety_norm.png'))
        
        print(f"Example images saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help="Path to input dataset (directory or file). If not provided, random noise is used.")
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of samples (or max images) to process")
    parser.add_argument('--resnet_checkpoint', type=str, default=r'E:\phd\1pppd\code\NudeNet\nsfw-resnet-master\checkpoints_nudenet_resnet50\model_138_50.pth', help="Path to ResNet checkpoint")
    parser.add_argument('--safety_model_path', type=str, default=r"E:\phd\4\code\VINE\CompVis\stable-diffusion-safety-checker", help="Path to Safety Checker model")
    parser.add_argument('--output_dir', type=str, default='spectral_results')
    parser.add_argument('--epsilon', type=float, default=32/255.0)
    parser.add_argument('--alpha', type=float, default=1/255.0)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    main(args)



# set PYTHONPATH=%cd%
# set TORCH_HOME=%cd%
# python vine/src/our/spectral_analysis.py ^
#   --num_samples 100 ^
#   --resnet_checkpoint "E:\phd\1pppd\code\NudeNet\nsfw-resnet-master\checkpoints_nudenet_resnet50\model_138_50.pth" ^
#   --safety_model_path "CompVis/stable-diffusion-safety-checker" ^
#   --output_dir spectral_results_random ^
