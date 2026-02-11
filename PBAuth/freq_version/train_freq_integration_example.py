"""
train_input01_o.py 中频域损失集成示例
放在文件顶部的导入部分
"""

# 在 train_input01_o.py 顶部的 import 部分添加：
from vine.src.Alias.freq_version.freq_loss_utils import FreqLossComputer, get_freq_stats

# ============================================================
# 在 main() 函数中，初始化频域损失计算器（约在第 57 行）
# ============================================================

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # [新增] 初始化频域损失计算器
    freq_loss_computer = FreqLossComputer(device=accelerator.device)

    watermark_encoder = VINE_Turbo(r"E:\phd\4\code\VINE\vine\src\output\finetuning_porn1127\checkpoint-50000")
    
    # ... 其他初始化代码 ...


# ============================================================
# 在训练循环中使用频域损失（约在第 400+ 行，where computed losses）
# ============================================================

                    # 原有代码：计算图像损失、LPIPS损失、秘密损失
                    image_loss = torch.mean((im_diff) ** 2)
                    lpips_loss = torch.mean(net_lpips(img_a, encoded_image))
                    
                    # [新增] 计算频域损失
                    # 注意：输入是 encoded_image（水印图） - img_a（原图） = residual
                    watermark_residual = encoded_image - img_a  # [B, 3, H, W]
                    
                    if global_step > args.freq_constraint_start_step:
                        # 使用综合频域损失
                        freq_loss, freq_loss_dict = freq_loss_computer.compute_combined_freq_loss(
                            residual=watermark_residual,
                            target_freq='low',  # 推动能量到低频
                            lf_radius=args.lf_radius,           # 默认 10
                            mf_radius=args.mf_radius,           # 默认 40
                            energy_weight=1.0,
                            smoothness_weight=0.1
                        )
                    else:
                        freq_loss = torch.tensor(0.0, device=accelerator.device)
                        freq_loss_dict = {'freq_energy_loss': 0.0, 'freq_smooth_loss': 0.0, 'freq_total_loss': 0.0}
                    
                    # 原有秘密损失计算
                    bce_logits_loss = torch.nn.BCEWithLogitsLoss().to(accelerator.device)
                    secret_loss = bce_logits_loss(nsfw_logits.unsqueeze(1), secret.float())
                    
                    # [修改] 总损失加入频域项（有动态权重）
                    if global_step > args.freq_constraint_start_step:
                        # 从 0 线性增加到 freq_loss_scale
                        freq_loss_scale_current = min(
                            args.freq_loss_scale * (global_step - args.freq_constraint_start_step) / 10000,
                            args.freq_loss_scale
                        )
                    else:
                        freq_loss_scale_current = 0.0
                    
                    if no_im_loss:
                        loss = secret_loss
                    else:
                        loss = (
                            loss_scales[0] * image_loss +
                            loss_scales[1] * lpips_loss +
                            loss_scales[2] * secret_loss +
                            freq_loss_scale_current * freq_loss  # [新增]
                        )
                        if not args.no_gan:
                            loss += loss_scales[3] * G_loss
                    
                    # 反向传播
                    accelerator.backward(loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                    
                    optimizer_gen.step()
                    lr_scheduler_gen.step()
                    optimizer_gen.zero_grad()
                    
                    # ... GAN 判别器训练代码 ...


# ============================================================
# 在日志记录部分加入频域相关指标（约在第 500+ 行）
# ============================================================

                    logs = {}
                    logs["train_chart/loss"] = loss.detach().item()
                    logs["train_chart/image_loss"] = image_loss.detach().item()
                    logs["train_chart/lpips_loss"] = lpips_loss.detach().item()
                    logs["train_chart/secret_loss"] = secret_loss.detach().item()
                    logs["train_chart/str_acc"] = str_acc
                    logs["train_chart/psnr"] = avg_psnr
                    
                    # [新增] 频域损失和统计
                    if global_step > args.freq_constraint_start_step:
                        logs["train_chart/freq_loss"] = freq_loss_dict['freq_total_loss']
                        logs["train_chart/freq_energy_loss"] = freq_loss_dict['freq_energy_loss']
                        logs["train_chart/freq_smooth_loss"] = freq_loss_dict['freq_smooth_loss']
                        
                        # 实时频域能量统计
                        freq_stats = get_freq_stats(
                            watermark_residual,
                            lf_radius=args.lf_radius,
                            mf_radius=args.mf_radius
                        )
                        logs["freq_stats/lf_pct"] = freq_stats['lf_pct']
                        logs["freq_stats/mf_pct"] = freq_stats['mf_pct']
                        logs["freq_stats/hf_pct"] = freq_stats['hf_pct']


# ============================================================
# 在验证阶段也添加频域统计（约在第 600+ 行）
# ============================================================

                        # 验证循环中
                        with torch.no_grad():
                            for i in tqdm(range(len(val_img)), desc="Validating", ncols=100):
                                # ... 验证代码 ...
                                
                                # [新增] 在验证结果中加入频域信息
                                val_residual = encoded_image - input_image
                                val_freq_stats = get_freq_stats(
                                    val_residual,
                                    lf_radius=args.lf_radius,
                                    mf_radius=args.mf_radius
                                )
                                
                        logs["val_chart/freq_lf_pct"] = val_freq_stats['lf_pct']
                        logs["val_chart/freq_mf_pct"] = val_freq_stats['mf_pct']
                        logs["val_chart/freq_hf_pct"] = val_freq_stats['hf_pct']
