import torch
import torch.nn.functional as F
import argparse
import os
import sys
import time
import io
from contextlib import redirect_stdout
try:
    import psutil
except ImportError:
    psutil = None
try:
    import gradio as gr
except ImportError:
    gr = None

from PIL import Image
from torchvision import transforms
import numpy as np
from transformers import CLIPImageProcessor
from vine.src.Alias.training_src.transformations import TransformNet
try:
    import lpips
except ImportError:
    lpips = None
    print("Warning: Could not import lpips. LPIPS loss will be disabled if requested.")

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    psnr = None
    ssim = None
    print("Warning: skimage not found. PSNR/SSIM will be disabled.")

try:
    from vine.src.Alias.other.color_l2 import ciede2000_loss
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'other'))
        from color_l2 import ciede2000_loss
    except ImportError:
        print("Warning: Could not import ciede2000_loss. Color loss will be disabled.")
        ciede2000_loss = None

# 尝试导入 PrivateDifferentiableSafetyChecker
# 优先尝试从 vine 包导入（如果 PYTHONPATH 设置正确）
try:
    from vine.src.Alias.private_safety_checker import PrivateDifferentiableSafetyChecker
except ImportError:
    # 如果脚本直接在当前目录下运行
    try:
        from private_safety_checker import PrivateDifferentiableSafetyChecker
    except ImportError:
        print("Error: Could not import PrivateDifferentiableSafetyChecker.")
        print("Please ensure you are running this script with the correct PYTHONPATH (e.g., set PYTHONPATH=%cd%).")
        sys.exit(1)


####set PYTHONPATH=E:\phd\4\code\VINE

def get_parser():
    parser = argparse.ArgumentParser(description="PGD Attack to trigger Safety Checker (Invisible Perturbation)")
    parser.add_argument("--web_ui", action="store_true", help="Launch Gradio Web UI")
    parser.add_argument("--input_path", type=str, default= r'E:\phd\4\code\VINE\vine\src\Alias\example\1.png', help="Path to input benign image")
    parser.add_argument("--output_path", type=str, default=r"E:\phd\4\code\VINE\vine\src\Alias\example\1_protected.png", help="Path to save adversarial image")
    parser.add_argument("--model_path", type=str, default=r"E:\phd\4\code\VINE\CompVis\stable-diffusion-safety-checker", help="Path to safety checker model")
    parser.add_argument("--epsilon", type=float, default=32/255.0, help="Perturbation budget (L_inf norm)")
    parser.add_argument("--alpha", type=float, default=2/255.0, help="Step size")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps")
    parser.add_argument("--color_loss_weight", type=float, default=0.5, help="Weight for CIEDE2000 color loss")
    parser.add_argument("--lpips_loss_weight", type=float, default=0, help="Weight for LPIPS loss (VGG)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 攻击参数
    parser.add_argument("--use_robust_attack", action="store_true", default=True, help="Whether to use robust attack (add noise during optimization)")
    
    # TransformNet 参数
    parser.add_argument("--imagenetc_step", type=int, default=160) # Disable by default for PGD
    parser.add_argument("--crop_resize_step", type=int, default=160) # Disable by default for PGD
    parser.add_argument("--ig_filter_step", type=int, default=160) # Disable by default for PGD
    
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
    return parser

def run_pgd_attack(args):
    start_time = time.time()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} not found.")
        return

    # 1. 加载 Safety Checker
    print(f"Loading Safety Checker from {args.model_path}...")
    try:
        # 使用我们自定义的可微 Safety Checker
        safety_checker = PrivateDifferentiableSafetyChecker.from_pretrained(args.model_path)
        feature_extractor = CLIPImageProcessor.from_pretrained(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model path is correct and points to a valid Stable Diffusion Safety Checker.")
        return

    safety_checker.to(args.device)
    safety_checker.eval()
    
    # 冻结模型参数
    for param in safety_checker.parameters():
        param.requires_grad = False

    # Initialize LPIPS model if needed
    loss_fn_lpips = None
    if args.lpips_loss_weight > 0:
        if lpips is not None:
            print("Loading LPIPS (VGG) model...")
            loss_fn_lpips = lpips.LPIPS(net='vgg').to(args.device)
            loss_fn_lpips.eval()
            for param in loss_fn_lpips.parameters():
                param.requires_grad = False
        else:
            print("Warning: LPIPS requested but library not found. Skipping LPIPS loss.")

    # 初始化 TransformNet 用于模拟攻击
    # 注意：这里我们初始化一个包含所有可能变换的 TransformNet，但具体启用哪些由 args 控制
    transform_net = TransformNet(
        device=args.device,
        rnd_bri=args.rnd_bri, rnd_hue=args.rnd_hue, rnd_sat=args.rnd_sat, rnd_noise=args.rnd_noise,
        do_jpeg=not args.no_jpeg, jpeg_quality=args.jpeg_quality,
        ic_up_level_interval=20 # 缩短间隔以便在 PGD 的短步数内生效
    ).to(args.device)
    # 激活 TransformNet
    transform_net.step0.fill_(1) 

    # 2. 加载并预处理图像
    img = Image.open(args.input_path).convert("RGB")
    
    # 获取 CLIP mean/std 用于后续归一化
    # inputs = feature_extractor(images=img, return_tensors="pt") # 原来的方式会 resize 到 224
    
    # 使用 torchvision transforms 加载图像，保持原始尺寸
    input_tensor = transforms.ToTensor()(img).unsqueeze(0).to(args.device) # [1, 3, H, W]
    
    # CLIP mean and std
    mean = torch.tensor(feature_extractor.image_mean).view(1, 3, 1, 1).to(args.device)
    std = torch.tensor(feature_extractor.image_std).view(1, 3, 1, 1).to(args.device)
    
    # 原始图像 Tensor [0, 1]
    original_image_tensor = input_tensor
    
    # 初始化对抗样本
    adv_image_tensor = original_image_tensor.clone().detach()
    
    print(f"Starting PGD attack...")
    print(f"  Epsilon: {args.epsilon:.4f} (approx {int(args.epsilon*255)} pixel levels)")
    print(f"  Steps: {args.steps}")
    print(f"  Alpha: {args.alpha:.4f}")
    if args.use_robust_attack:
        print(f"  Robust Attack Enabled: Progressive noise addition enabled.")

    # 3. 攻击循环
    for i in range(args.steps):
        adv_image_tensor.requires_grad = True
        
        # --- 鲁棒性攻击模拟 (Progressive) ---
        if args.use_robust_attack:
            # 1. 扩展到 [-1, 1] 以适应 TransformNet
            adv_input_trans = adv_image_tensor * 2.0 - 1.0
            
            # 2. 构造模拟训练过程的 args
            # 我们将 PGD 的 steps 映射到训练的 global_step
            # 假设 PGD 10000 步对应训练的 20000 步 (或者根据需要调整比例)
            # 这样可以让噪声强度随 PGD 迭代逐渐增加
            simulated_global_step = int(i * (200 / args.steps)) 
            
            # 应用变换
            # TransformNet 内部会根据 global_step 计算 ramp_fn，从而逐步增加强度
            attacked_image = transform_net(adv_input_trans, adv_input_trans, global_step=simulated_global_step, args=args)
            
            # 3. 恢复到 [0, 1]
            attacked_image = (attacked_image + 1.0) / 2.0
            
            # 4. Resize to 224x224 for Safety Checker and Normalize
            model_input_raw = F.interpolate(attacked_image, size=(224, 224), mode='bilinear', align_corners=False)
            model_input = (model_input_raw - mean) / std
        else:
            # 不加攻击，直接输入
            # Resize to 224x224 for Safety Checker and Normalize
            model_input_raw = F.interpolate(adv_image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            model_input = (model_input_raw - mean) / std
        
        # Debugging gradients
        # print(f"Step {i}: adv_image_tensor.requires_grad = {adv_image_tensor.requires_grad}")
        # if args.use_robust_attack:
        #     print(f"  simulated_global_step = {simulated_global_step}")
        #     print(f"  attacked_image.requires_grad = {attacked_image.requires_grad}")
        # print(f"  model_input.requires_grad = {model_input.requires_grad}")

        # 前向传播
        # 我们希望触发 Safety Checker => 最大化 nsfw_logits
        # Loss = -nsfw_logits
        _, nsfw_logits, nsfw_probs, _ = safety_checker(clip_input=model_input, nsfw_threshold=0.0)
        # print(f"  nsfw_logits.requires_grad = {nsfw_logits.requires_grad}")
        
        loss_nsfw = -nsfw_logits.mean() 
        
        # Calculate Color Loss (CIEDE2000)
        loss_color = torch.tensor(0.0, device=args.device)
        if args.color_loss_weight > 0 and ciede2000_loss is not None:
            # ciede2000_loss expects inputs in [0, 1]
            loss_color = ciede2000_loss(adv_image_tensor, original_image_tensor).mean()
            
        # Calculate LPIPS Loss
        loss_lpips = torch.tensor(0.0, device=args.device)
        if args.lpips_loss_weight > 0 and loss_fn_lpips is not None:
            # LPIPS expects inputs in [-1, 1]
            adv_norm = adv_image_tensor * 2.0 - 1.0
            orig_norm = original_image_tensor * 2.0 - 1.0
            loss_lpips = loss_fn_lpips(adv_norm, orig_norm).mean()

        total_loss =    loss_nsfw + args.color_loss_weight * loss_color + args.lpips_loss_weight * loss_lpips
        # print(f"  loss.requires_grad = {loss.requires_grad}")
        
        safety_checker.zero_grad()
        total_loss.backward()
        
        grad = adv_image_tensor.grad.data
        
        # PGD 更新 (沿着梯度方向增加 Logits, 减小 Color/LPIPS Loss)
        
        adv_image_tensor = adv_image_tensor.detach() - args.alpha * grad.sign()
        
        # 投影到 Epsilon 球 (Projection)
        delta = adv_image_tensor - original_image_tensor
        delta = torch.clamp(delta, -args.epsilon, args.epsilon)
        adv_image_tensor = torch.clamp(original_image_tensor + delta, 0.0, 1.0)
        
        if i % 10 == 0 or i == args.steps - 1:
            print(f"Step {i:3d}: Total={total_loss.item():.4f} (NSFW={loss_nsfw.item():.4f}, Color={loss_color.item():.4f}, LPIPS={loss_lpips.item():.4f}), Prob={nsfw_probs.item():.4f}")
            if nsfw_probs.item() > 0.5:
                print(f"  -> Threshold crossed at step {i}!")

    # 4. 最终评估与保存
    with torch.no_grad():
        # 最终评估时，我们也可以看看在有攻击和无攻击下的表现
        print("-" * 30)
        print(f"Final Result (Clean):")
        # Resize for evaluation
        final_input_clean_raw = F.interpolate(adv_image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        final_input_clean = (final_input_clean_raw - mean) / std
        _, _, final_probs_clean, final_check_clean = safety_checker(clip_input=final_input_clean)
        print(f"  NSFW Probability: {final_probs_clean.item():.4f}")
        print(f"  Triggered: {final_check_clean.item()}")
        
        if args.use_robust_attack:
            print(f"Final Result (Robust - Simulated):")
            # 再次模拟一次攻击来评估 (使用最大强度)
            adv_input_trans = adv_image_tensor * 2.0 - 1.0
            
            # 使用一个较大的 step 来模拟最终的强攻击 (approx 20000 as used in loop)
            final_simulated_step = 200
            
            attacked_image_final = transform_net(adv_input_trans, adv_input_trans, global_step=final_simulated_step, args=args)
            
            attacked_image_final = (attacked_image_final + 1.0) / 2.0
            
            # Resize for evaluation
            final_input_robust_raw = F.interpolate(attacked_image_final, size=(224, 224), mode='bilinear', align_corners=False)
            final_input_robust = (final_input_robust_raw - mean) / std
            _, _, final_probs_robust, final_check_robust = safety_checker(clip_input=final_input_robust)
            print(f"  NSFW Probability: {final_probs_robust.item():.4f}")
            print(f"  Triggered: {final_check_robust.item()}")

    # 保存图像
    adv_image_np = adv_image_tensor.squeeze(0).cpu().numpy()
    adv_image_np = np.transpose(adv_image_np, (1, 2, 0)) # HWC
    adv_image_np = (adv_image_np * 255).astype(np.uint8)
    adv_pil = Image.fromarray(adv_image_np)
    adv_pil.save(args.output_path)
    print(f"Saved adversarial image to {args.output_path}")

    # 5. Image Quality Evaluation
    print("-" * 30)
    print("Image Quality Metrics:")
    
    # Prepare images for metrics (numpy, uint8, HWC)
    orig_np = original_image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    orig_np = (orig_np * 255).astype(np.uint8)
    
    adv_np = adv_image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    adv_np = (adv_np * 255).astype(np.uint8)
    
    if psnr is not None and ssim is not None:
        p_val = psnr(orig_np, adv_np)
        # ssim requires channel_axis for multichannel
        s_val = ssim(orig_np, adv_np, channel_axis=2)
        print(f"  PSNR: {p_val:.4f} dB")
        print(f"  SSIM: {s_val:.4f}")
        
    if lpips is not None:
        # Use AlexNet for evaluation as it's the standard metric
        try:
            # Suppress print from LPIPS loading if possible, or just load
            loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).to(args.device)
            loss_fn_alex.eval()
            # LPIPS expects [-1, 1]
            orig_norm = original_image_tensor * 2.0 - 1.0
            adv_norm = adv_image_tensor * 2.0 - 1.0
            with torch.no_grad():
                d_val = loss_fn_alex(orig_norm, adv_norm)
            print(f"  LPIPS (Alex): {d_val.item():.4f}")
        except Exception as e:
            print(f"  LPIPS (Alex) calculation failed: {e}")

    end_time = time.time()
    print("="*40)
    print(f"Total Runtime: {end_time - start_time:.4f} seconds")
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    if psutil:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        if hasattr(mem_info, 'peak_wset'):
            print(f"Peak System Memory (Peak WSet): {mem_info.peak_wset / 1024 / 1024:.2f} MB")
        else:
            print(f"Current System Memory (RSS): {mem_info.rss / 1024 / 1024:.2f} MB")
    print("="*40)
    return adv_pil

def launch_gradio_ui(default_args):
    if gr is None:
        print("Error: Gradio is not installed. Please install it with 'pip install gradio'.")
        return

    def attack_wrapper(input_img, model_path, epsilon, alpha, steps, 
                       color_weight, lpips_weight, use_robust, 
                       rnd_noise, contrast_low, contrast_high, rnd_sat, rnd_bri, rnd_hue, jpeg_quality,
                       no_motionBlur, no_gaussianNoise, no_contrast, no_bright, no_saturation, no_jpeg):
        
        # Create a temporary path for input image if it's a numpy array or PIL
        # Gradio 'filepath' type passes a str.
        if input_img is None:
            return None, "Please upload an image."
            
        # Update default_args with new values
        # We create a new Namespace to avoid modifying the global default_args
        args = argparse.Namespace(**vars(default_args))
        args.input_path = input_img
        args.model_path = model_path
        args.epsilon = epsilon
        args.alpha = alpha
        args.steps = int(steps)
        args.color_loss_weight = color_weight
        args.lpips_loss_weight = lpips_weight
        args.use_robust_attack = use_robust
        
        # TransformNet params
        args.rnd_noise = rnd_noise
        args.contrast_low = contrast_low
        args.contrast_high = contrast_high
        args.rnd_sat = rnd_sat
        args.rnd_bri = rnd_bri
        args.rnd_hue = rnd_hue
        args.jpeg_quality = int(jpeg_quality)
        
        args.no_motionBlur = no_motionBlur
        args.no_gaussianNoise = no_gaussianNoise
        args.no_contrast = no_contrast
        args.no_bright = no_bright
        args.no_saturation = no_saturation
        args.no_jpeg = no_jpeg
        
        # Generate a temporary output path
        args.output_path = os.path.join(os.path.dirname(input_img), "gradio_output.png")
        
        # Capture stdout
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                adv_pil = run_pgd_attack(args)
                logs = f.getvalue()
                return adv_pil, logs
            except Exception as e:
                print(f"Error during attack: {e}")
                logs = f.getvalue()
                return None, logs

    with gr.Blocks(title="PGD Attack Visualization") as demo:
        gr.Markdown("# PGD Attack Visualization Tool")
        gr.Markdown("Generate adversarial examples to trigger the Stable Diffusion Safety Checker.")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Input Image")
                
                with gr.Accordion("Model & Attack Parameters", open=True):
                    model_path = gr.Textbox(value=default_args.model_path, label="Safety Checker Model Path")
                    epsilon = gr.Slider(0, 1, value=default_args.epsilon, label="Epsilon (Perturbation Budget)")
                    alpha = gr.Slider(0, 0.1, value=default_args.alpha, label="Alpha (Step Size)")
                    steps = gr.Slider(1, 1000, value=default_args.steps, step=1, label="Steps")
                
                with gr.Accordion("Loss Weights", open=True):
                    color_weight = gr.Slider(0, 10, value=default_args.color_loss_weight, label="Color Loss Weight (CIEDE2000)")
                    lpips_weight = gr.Slider(0, 10, value=default_args.lpips_loss_weight, label="LPIPS Loss Weight")
                
                with gr.Accordion("Robustness Settings", open=False):
                    use_robust = gr.Checkbox(value=default_args.use_robust_attack, label="Use Robust Attack (TransformNet)")
                    
                    with gr.Group():
                        gr.Markdown("### TransformNet Parameters")
                        rnd_noise = gr.Slider(0, 0.1, value=default_args.rnd_noise, label="Random Noise")
                        with gr.Row():
                            contrast_low = gr.Slider(0, 1, value=default_args.contrast_low, label="Contrast Low")
                            contrast_high = gr.Slider(1, 2, value=default_args.contrast_high, label="Contrast High")
                        
                        with gr.Row():
                            rnd_sat = gr.Slider(0, 2, value=default_args.rnd_sat, label="Random Saturation")
                            rnd_bri = gr.Slider(0, 1, value=default_args.rnd_bri, label="Random Brightness")
                            rnd_hue = gr.Slider(0, 0.5, value=default_args.rnd_hue, label="Random Hue")
                        
                        jpeg_quality = gr.Slider(1, 100, value=default_args.jpeg_quality, step=1, label="JPEG Quality")
                        
                        gr.Markdown("### Disable Augmentations")
                        with gr.Row():
                            no_motionBlur = gr.Checkbox(value=default_args.no_motionBlur, label="No Motion Blur")
                            no_gaussianNoise = gr.Checkbox(value=default_args.no_gaussianNoise, label="No Gaussian Noise")
                            no_contrast = gr.Checkbox(value=default_args.no_contrast, label="No Contrast")
                        with gr.Row():
                            no_bright = gr.Checkbox(value=default_args.no_bright, label="No Brightness")
                            no_saturation = gr.Checkbox(value=default_args.no_saturation, label="No Saturation")
                            no_jpeg = gr.Checkbox(value=default_args.no_jpeg, label="No JPEG")

                run_btn = gr.Button("Run Attack", variant="primary")
            
            with gr.Column(scale=1):
                output_image = gr.Image(label="Adversarial Image")
                logs = gr.Textbox(label="Logs & Metrics", lines=20)

        run_btn.click(
            fn=attack_wrapper,
            inputs=[
                input_image, model_path, epsilon, alpha, steps,
                color_weight, lpips_weight, use_robust,
                rnd_noise, contrast_low, contrast_high, rnd_sat, rnd_bri, rnd_hue, jpeg_quality,
                no_motionBlur, no_gaussianNoise, no_contrast, no_bright, no_saturation, no_jpeg
            ],
            outputs=[output_image, logs]
        )

    demo.launch(share=True)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if args.web_ui:
        launch_gradio_ui(args)
    else:
        run_pgd_attack(args)

if __name__ == "__main__":
    main()