"""
频域正则化损失函数模块
用于约束水印能量集中在指定频域（低/中/高频）
"""

import torch
import torch.nn.functional as F
import numpy as np

def loss_lowfreq(residual, lf_radius=12, mf_radius=40, weight=0.2):
    """
    惩罚中高频能量，使模型主动把信息压到 LF
    """
    B, C, H, W = residual.shape
    fft = torch.fft.fftshift(torch.fft.fft2(residual))
    mag = torch.abs(fft)

    cy, cx = H//2, W//2
    Y, X = torch.meshgrid(
        torch.arange(H, device=residual.device),
        torch.arange(W, device=residual.device),
        indexing='ij')
    dist = ((X - cx)**2 + (Y - cy)**2).sqrt()

    mask_hf = dist > mf_radius
    mask_mf = (dist > lf_radius) & (dist <= mf_radius)

    hf_energy = (mag[:, :, mask_hf]**2).mean()
    mf_energy = (mag[:, :, mask_mf]**2).mean()

    return weight * (hf_energy + mf_energy)


class FreqLossComputer:
    """
    高效计算频域正则化损失
    支持批处理，梯度传播友好
    """
    
    def __init__(self, device='cpu', fft_cache_size=256):
        self.device = device
        self.fft_cache = {}  # 缓存频域掩码，避免重复计算
        self.fft_cache_size = fft_cache_size
    
    def _get_freq_mask(self, H, W, lf_radius=10, mf_radius=40, mode='low'):
        """
        生成频域掩码（缓存）
        mode: 'low' / 'mid' / 'high' / 'high_only'
        """
        cache_key = (H, W, lf_radius, mf_radius, mode)
        if cache_key in self.fft_cache:
            return self.fft_cache[cache_key]
        
        # 构造距离矩阵
        cy, cx = H // 2, W // 2
        Y, X = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        dist = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        
        if mode == 'low':
            mask = dist <= lf_radius
        elif mode == 'mid':
            mask = (dist > lf_radius) & (dist <= mf_radius)
        elif mode == 'high':
            mask = dist > mf_radius
        elif mode == 'high_only':
            # 只保留最高频部分（>= mf_radius）
            mask = dist >= mf_radius
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 缓存结果
        if len(self.fft_cache) < self.fft_cache_size:
            self.fft_cache[cache_key] = mask
        
        return mask
    
    def compute_freq_energy_loss(
        self,
        residual,
        target_freq='low',
        lf_radius=10,
        mf_radius=40,
        energy_weight=None
    ):
        """
        约束水印残差能量集中在目标频域
        
        Args:
            residual: [B, C, H, W] 水印残差 (encoded_image - orig_image)
            target_freq: 目标频域 ('low' / 'mid' / 'high')
            lf_radius: 低频半径 (像素)
            mf_radius: 中频半径 (像素)
            energy_weight: 能量加权系数 (None 表示均匀权重)
        
        Returns:
            loss: 标量，梯度友好
        """
        B, C, H, W = residual.shape
        
        # 2D FFT（对每个 batch 和 channel）
        # fft2d: [B, C, H, W] -> 复数张量
        fft = torch.fft.fftshift(torch.fft.fft2(residual))
        mag = torch.abs(fft)
        
        # 获取目标频域掩码
        if target_freq == 'low':
            # 最小化高频能量 -> 推动能量到低频
            mask_target = self._get_freq_mask(H, W, lf_radius, mf_radius, mode='low')
            mask_unwanted = self._get_freq_mask(H, W, lf_radius, mf_radius, mode='high')
        elif target_freq == 'mid':
            # 最小化低频和高频 -> 推动能量到中频
            mask_target = self._get_freq_mask(H, W, lf_radius, mf_radius, mode='mid')
            mask_unwanted = ~mask_target
        elif target_freq == 'high':
            # 最小化低频和中频 -> 推动能量到高频（用于对比）
            mask_target = self._get_freq_mask(H, W, lf_radius, mf_radius, mode='high')
            mask_unwanted = ~mask_target
        else:
            raise ValueError(f"Unknown target_freq: {target_freq}")
        
        # 计算目标频域能量
        energy_target = mag[:, :, mask_target].sum(dim=-1)  # [B, C]
        energy_total = mag.sum(dim=(-2, -1))  # [B, C]
        
        # 防止除零
        energy_total = torch.clamp(energy_total, min=1e-8)
        
        # 能量比例 [B, C]
        energy_ratio = energy_target / energy_total  # 目标频域能量占比
        
        # 策略 1: 最大化目标频域能量占比（推荐）
        # 目标是让目标频域占比尽可能高
        loss = 1.0 - energy_ratio.mean()
        
        return loss
    
    def compute_freq_smoothness_loss(
        self,
        residual,
        target_freq='low',
        lf_radius=10,
        mf_radius=40,
        smoothness_strength=1.0
    ):
        """
        平滑约束：在目标频域内加强相邻频率成分的一致性
        （有助于防止孤立高频峰值）
        
        Args:
            residual: [B, C, H, W] 水印残差
            target_freq: 目标频域
            smoothness_strength: 平滑强度系数
        
        Returns:
            loss: 标量
        """
        B, C, H, W = residual.shape
        
        # FFT
        fft = torch.fft.fftshift(torch.fft.fft2(residual))
        mag = torch.abs(fft)
        
        # 获取目标掩码
        if target_freq == 'low':
            mask = self._get_freq_mask(H, W, lf_radius, mf_radius, mode='low')
        else:
            mask = self._get_freq_mask(H, W, lf_radius, mf_radius, mode=target_freq)
        
        # 在目标频域内，计算相邻像素梯度
        # 将掩码外的能量置零
        mag_masked = mag.clone()
        mag_masked[:, :, ~mask] = 0
        
        # 计算梯度（差分）
        grad_h = torch.abs(mag_masked[:, :, 1:, :] - mag_masked[:, :, :-1, :])
        grad_w = torch.abs(mag_masked[:, :, :, 1:] - mag_masked[:, :, :, :-1])
        
        # 平均梯度
        loss = (grad_h.mean() + grad_w.mean()) * smoothness_strength
        
        return loss
    
    def compute_combined_freq_loss(
        self,
        residual,
        target_freq='low',
        lf_radius=10,
        mf_radius=40,
        energy_weight=1.0,
        smoothness_weight=0.1
    ):
        """
        综合频域损失 = 能量集中 + 平滑约束
        
        Args:
            residual: [B, C, H, W] 水印残差
            target_freq: 目标频域
            energy_weight: 能量项权重
            smoothness_weight: 平滑项权重
        
        Returns:
            loss: 总损失
            loss_dict: 损失分项字典
        """
        loss_energy = self.compute_freq_energy_loss(
            residual, target_freq, lf_radius, mf_radius
        )
        
        loss_smooth = self.compute_freq_smoothness_loss(
            residual, target_freq, lf_radius, mf_radius
        )
        
        total_loss = energy_weight * loss_energy + smoothness_weight * loss_smooth
        
        return total_loss, {
            'freq_energy_loss': loss_energy.item(),
            'freq_smooth_loss': loss_smooth.item(),
            'freq_total_loss': total_loss.item()
        }


def get_freq_stats(residual, lf_radius=10, mf_radius=40):
    """
    快速获取频域能量统计（用于监控）
    
    Args:
        residual: [B, C, H, W]
        lf_radius, mf_radius: 频域半径
    
    Returns:
        dict: 包含各频域能量占比
    """
    B, C, H, W = residual.shape
    
    fft = torch.fft.fftshift(torch.fft.fft2(residual))
    mag = torch.abs(fft)
    
    cy, cx = H // 2, W // 2
    Y, X = torch.meshgrid(
        torch.arange(H, device=residual.device, dtype=torch.float32),
        torch.arange(W, device=residual.device, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    
    mask_lf = dist <= lf_radius
    mask_mf = (dist > lf_radius) & (dist <= mf_radius)
    mask_hf = dist > mf_radius
    
    energy_lf = mag[:, :, mask_lf].sum()
    energy_mf = mag[:, :, mask_mf].sum()
    energy_hf = mag[:, :, mask_hf].sum()
    energy_total = mag.sum()
    
    return {
        'lf_pct': (energy_lf / energy_total * 100).item(),
        'mf_pct': (energy_mf / energy_total * 100).item(),
        'hf_pct': (energy_hf / energy_total * 100).item(),
    }
