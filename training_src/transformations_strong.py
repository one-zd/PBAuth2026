# rewritten_transformnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import torchvision.transforms as transforms

# keep your existing imports if needed, e.g. extra_utils, RandomImagenetC, JpegCompression, MbrsNoise, etc.
import vine.src.Alias.training_src.extra_utils as extra_utils
from vine.src.Alias.training_src.augment_imagenetc import RandomImagenetC
from vine.src.Alias.training_src.other_noises import Cropout, Dropout, Resize
from vine.src.Alias.training_src.jpeg_compression import JpegCompression
from vine.src.Alias.training_src.mbrs_noise import MbrsNoise
import pilgram


def get_gaussian_kernel2d(kernel_size, sigma, device=None, dtype=torch.float32):
    """Return 2D gaussian kernel (kernel_size: (h, w), sigma: (sy, sx))."""
    ky, kx = kernel_size
    sy, sx = sigma
    y = torch.arange(-(ky // 2), ky // 2 + 1, dtype=dtype, device=device)
    x = torch.arange(-(kx // 2), kx // 2 + 1, dtype=dtype, device=device)
    yy = y.view(-1, 1)
    xx = x.view(1, -1)
    kernel = torch.exp(-0.5 * ((yy / sy) ** 2 + (xx / sx) ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def motion_kernel(length, angle=0.0, device=None, dtype=torch.float32):
    """Generate a simple motion blur kernel of size length x length rotated by angle (radians)."""
    # create horizontal line kernel then rotate by angle using affine grid
    if length <= 1:
        k = torch.zeros((1, 1), device=device, dtype=dtype)
        k[0, 0] = 1.0
        return k
    k = torch.zeros((length, length), device=device, dtype=dtype)
    center = length // 2
    k[center, :] = 1.0
    k = k / k.sum()
    # rotate kernel using grid_sample
    theta = torch.tensor([[ [np.cos(angle), -np.sin(angle), 0.0],
                            [np.sin(angle),  np.cos(angle), 0.0] ]], device=device, dtype=dtype)
    grid = F.affine_grid(theta, size=(1, 1, length, length), align_corners=False)
    k = k.view(1, 1, length, length)
    k = F.grid_sample(k, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    k = k.view(length, length)
    k = k / (k.sum() + 1e-12)
    return k


class StrongBlurModule:
    """
    Implements several strong blur transforms:
    - pixelate (downscale/upscale)
    - large gaussian / defocus-like (large kernel)
    - zoom blur (averaging different scaled versions)
    - motion blur with random angle and large kernel
    """
    def __init__(self, device='cpu', dtype=torch.float32, pixelate_prob=0.2,
                 defocus_prob=0.25, zoom_prob=0.2, motion_prob=0.25, large_gauss_prob=0.1):
        self.device = torch.device(device)
        self.dtype = dtype
        self.pixelate_prob = pixelate_prob
        self.defocus_prob = defocus_prob
        self.zoom_prob = zoom_prob
        self.motion_prob = motion_prob
        self.large_gauss_prob = large_gauss_prob

    def __call__(self, x):
        # x: [B, C, H, W], values in [0,1]
        r = random.random()
        if r < self.pixelate_prob:
            return self.pixelate(x)
        r -= self.pixelate_prob
        if r < self.defocus_prob:
            return self.defocus_blur(x)
        r -= self.defocus_prob
        if r < self.zoom_prob:
            return self.zoom_blur(x)
        r -= self.zoom_prob
        if r < self.motion_prob:
            return self.motion_blur(x)
        # fallback: large gaussian
        return self.large_gaussian(x)

    def pixelate(self, x, down_min=8, down_max=32):
        B, C, H, W = x.shape
        down = random.randint(down_min, down_max)
        # ensure at least 2 px
        dh = max(2, H // down)
        dw = max(2, W // down)
        small = F.interpolate(x, size=(dh, dw), mode='nearest')
        up = F.interpolate(small, size=(H, W), mode='nearest')
        return up

    def defocus_blur(self, x, radius_min=3, radius_max=11):
        B, C, H, W = x.shape
        radius = random.randint(radius_min, radius_max)
        ksize = radius * 2 + 1
        sigma = radius * 0.8 + 0.1
        kernel = get_gaussian_kernel2d((ksize, ksize), (sigma, sigma), device=x.device, dtype=x.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # 1x1xksizexksize
        kernel = kernel.repeat(C, 1, 1, 1)  # C x 1 x k x k for depthwise conv
        padding = ksize // 2
        x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        out = F.conv2d(x, kernel, groups=C)
        return out

    def large_gaussian(self, x, radius_min=7, radius_max=21):
        B, C, H, W = x.shape
        radius = random.randint(radius_min, radius_max)
        ksize = radius * 2 + 1
        sigma = max(0.5, radius * 0.5)
        kernel = get_gaussian_kernel2d((ksize, ksize), (sigma, sigma), device=x.device, dtype=x.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        padding = ksize // 2
        x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        out = F.conv2d(x, kernel, groups=C)
        return out

    def zoom_blur(self, x, num=6, max_scale=1.08):
        B, C, H, W = x.shape
        outs = []
        for i in range(num):
            scale = 1.0 + (i / float(num)) * (max_scale - 1.0)
            xs = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            xs = F.interpolate(xs, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(xs)
        out = sum(outs) / float(len(outs))
        return out

    def motion_blur(self, x, length_min=7, length_max=31):
        B, C, H, W = x.shape
        length = random.randint(length_min, length_max)
        if length % 2 == 0:
            length += 1
        angle = random.uniform(0, np.pi)
        k = motion_kernel(length, angle=angle, device=x.device, dtype=x.dtype)
        k = k.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)  # depthwise
        padding = length // 2
        x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        out = F.conv2d(x, k, groups=C)
        return out


class RewrittenTransformNet(nn.Module):
    def __init__(self, device, rnd_bri=0.3, rnd_hue=0.1, do_jpeg=False, jpeg_quality=50, rnd_noise=0.02,
                 rnd_sat=1.0, rnd_trans=0.1, contrast=(0.5, 1.5), ramp=1000, imagenetc_level=5,
                 ic_up_level_interval=10000, strong_blur_prob=0.5, enable_lowfreq_project=False) -> None:
        """
        Drop-in replacement for your TransformNet with strong blur family integrated.
        enable_lowfreq_project: if True, project residual to low-frequency mask (fast experiment tool).
        """
        super().__init__()
        self.device = device
        self.rnd_bri = rnd_bri
        self.rnd_hue = rnd_hue
        self.jpeg_quality = jpeg_quality
        self.rnd_noise = rnd_noise
        self.rnd_sat = rnd_sat
        self.rnd_trans = rnd_trans
        self.contrast_low, self.contrast_high = contrast
        self.do_jpeg = do_jpeg
        self.register_buffer('step0', torch.tensor(0))
        self.strong_blur_prob = strong_blur_prob
        self.enable_lowfreq_project = enable_lowfreq_project

        if imagenetc_level > 0:
            self.imagenetc = ImagenetCTransform(max_severity=imagenetc_level)

        self.cropout = Cropout()
        self.dropout = Dropout(keep_ratio_range=[0.7, 0.9])
        self.resize = Resize(resize_ratio_range=[0.5, 2.0])
        self.ig_filter = IG_Filter()

        self.jpeg = JpegCompression(device=device)
        self.up_level_interval = 2000
        self.ic_up_level_interval = ic_up_level_interval
        # keep existing MBRS usage
        self.mbrs_noise = MbrsNoise(['Combined([JpegMask(50),Jpeg(50)])'])

        # new strong blur module
        self.strong_blur = StrongBlurModule(device=device, dtype=torch.float32,)

    def activate(self, global_step):
        if self.step0 == 0:
            print(f'[TRAINING] Activating RewrittenTransformNet at step {global_step}')
            self.step0 = torch.tensor(global_step)

    def is_activated(self):
        return self.step0 > 0

    def maybe_lowfreq_project(self, residual, low_radius=12):
        """
        Optionally project residual to low-frequency by applying FFT, zeroing high-frequency rings,
        and inverse FFT. residual in spatial domain.
        Note: use sparingly (experimental).
        """
        # residual: [B, C, H, W] in [-1,1] or [0,1] depending caller. We'll handle in float.
        device = residual.device
        B, C, H, W = residual.shape
        # work on luminance or each channel
        fft = torch.fft.fft2(residual, norm='ortho')
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        # build lowpass mask
        yy = torch.arange(-H//2, H//2, device=device).view(-1, 1).repeat(1, W)
        xx = torch.arange(-W//2, W//2, device=device).view(1, -1).repeat(H, 1)
        rr = torch.sqrt((yy.float() ** 2) + (xx.float() ** 2))
        mask = (rr <= low_radius).float()  # H x W
        mask = mask.unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
        mask = mask.to(fft.dtype)
        fft = fft * mask
        fft = torch.fft.ifftshift(fft, dim=(-2, -1))
        img = torch.fft.ifft2(fft, norm='ortho').real
        return img

    def forward(self, encoded_image, cover_img, global_step, args, p=0.999):
        """
        encoded_image, cover_img: tensors in [-1,1], shape [B,3,H,W]
        returns same type as input (encoded_image dtype)
        """
        if torch.rand(1)[0] >= p:
            return encoded_image

        encoded_image_type = encoded_image.dtype
        encoded_image = encoded_image.to(torch.float32)
        cover_img = cover_img.to(torch.float32)

        # ImagenetC block (kept from original) - 降低概率给strong blur更多机会
        if hasattr(self, 'imagenetc') and torch.rand(1)[0] < 0.3 and global_step > getattr(args, 'imagenetc_step', 0):
            level = min(int((global_step - getattr(args, 'imagenetc_step', 0)) / self.ic_up_level_interval) + 1, 7)
            if global_step < 6 * self.ic_up_level_interval + getattr(args, 'imagenetc_step', 0):
                corrupt_strength = level
            else:
                corrupt_strength = np.random.randint(1, level + 1)
            encoded_image = self.imagenetc(encoded_image, corrupt_strength=corrupt_strength)
            return encoded_image.to(encoded_image_type)

        # 优先执行strong blur（主要变化）
        if torch.rand(1)[0] < self.strong_blur_prob:
            # apply strong blur on the encoded image (values in [0,1])
            encoded_image = self.strong_blur(encoded_image)
            encoded_image = torch.clamp(encoded_image, 0, 1)

        # rescale to [0,1]
        encoded_image = encoded_image * 0.5 + 0.5
        cover_img = cover_img * 0.5 + 0.5

        # progressive ramp helper
        ramp_fn = lambda ramp: np.min([(global_step - self.step0.cpu().item()) / ramp, 1.]) if ramp > 0 else 1.0

        # cropout/resize logic (kept as previous)
        if global_step > getattr(args, 'crop_resize_step', 0):
            # adaptively increase cropout intensity
            level = min(0.05 * (int((global_step - getattr(args, 'crop_resize_step', 0)) / self.up_level_interval) + 1), 0.5)
            if global_step < 6 * self.up_level_interval + getattr(args, 'crop_resize_step', 0):
                max_tamper_area = level
            else:
                max_tamper_area = np.random.uniform(0.1, 0.5)
            encoded_image = self.cropout([encoded_image, cover_img], max_tamper_area=max_tamper_area,
                                         height_ratio_range=(max(1 - max_tamper_area, 0.8), max(1 - max_tamper_area, 0.9)),
                                         width_ratio_range=(max(1 - max_tamper_area, 0.8), max(1 - max_tamper_area, 0.9)))[0]
            encoded_image = torch.clamp(encoded_image, 0, 1)
            encoded_image = encoded_image * 2 - 1
            return encoded_image

        # IG filter block (降低概率)
        if torch.rand(1)[0] < 0.2 and global_step > getattr(args, 'ig_filter_step', 0):
            encoded_image = self.ig_filter(encoded_image)
            encoded_image = torch.clamp(encoded_image, 0, 1)
            encoded_image = encoded_image * 2 - 1
            return encoded_image

        # lightweight blur (existing small-kernel) - keep but less effect
        if not getattr(args, 'no_motionBlur', False):
            N_blur = getattr(args, 'N_blur', 3)
            f = extra_utils.random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                               wmin_line=3)
            f = f.to(encoded_image.device, encoded_image.dtype)
            encoded_image = F.conv2d(encoded_image, f, bias=None, padding=int((N_blur - 1) / 2))

        # additive gaussian noise (kept)
        rnd_noise = torch.rand(1)[0] * ramp_fn(getattr(args, 'rnd_noise_ramp', 1)) * getattr(args, 'rnd_noise', self.rnd_noise)
        if not getattr(args, 'no_gaussianNoise', False):
            noise = torch.normal(mean=0, std=float(rnd_noise), size=encoded_image.size(), dtype=encoded_image.dtype, device=encoded_image.device)
            encoded_image = encoded_image + noise
            encoded_image = torch.clamp(encoded_image, 0, 1)

        # contrast & brightness & saturation (kept)
        rnd_bri = ramp_fn(getattr(args, 'rnd_bri_ramp', 1)) * getattr(args, 'rnd_bri', self.rnd_bri)
        rnd_hue = ramp_fn(getattr(args, 'rnd_hue_ramp', 1)) * getattr(args, 'rnd_hue', self.rnd_hue)
        contrast_low = 1. - (1. - getattr(args, 'contrast_low', self.contrast_low)) * ramp_fn(getattr(args, 'contrast_ramp', 1))
        contrast_high = 1. + (getattr(args, 'contrast_high', self.contrast_high) - 1.) * ramp_fn(getattr(args, 'contrast_ramp', 1))
        contrast_params = [contrast_low, contrast_high]
        rnd_sat = torch.rand(1)[0] * ramp_fn(getattr(args, 'rnd_sat_ramp', 1)) * getattr(args, 'rnd_sat', self.rnd_sat)

        rnd_brightness = extra_utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, encoded_image.shape[0])
        contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(contrast_params[0], contrast_params[1]).to(encoded_image.device).reshape(encoded_image.size()[0], 1, 1, 1)
        if not getattr(args, 'no_contrast', False):
            encoded_image = encoded_image * contrast_scale
        if not getattr(args, 'no_bright', False):
            encoded_image = encoded_image + rnd_brightness.to(encoded_image.device)
        encoded_image = torch.clamp(encoded_image, 0, 1)

        # saturation
        sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1).to(encoded_image.device)
        if not getattr(args, 'no_saturation', False):
            encoded_image_lum = torch.sum(encoded_image * sat_weight, dim=1, keepdim=True)
            encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

        # JPEG / MBRS as in original (kept, but call probability updated)
        if global_step < 10000:
            jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(getattr(args, 'jpeg_quality_ramp', 1)) * (100. - getattr(args, 'jpeg_quality', self.jpeg_quality))
            if jpeg_quality < 50:
                jpeg_factor = 5000. / jpeg_quality
            else:
                jpeg_factor = 200. - jpeg_quality * 2
            jpeg_factor = jpeg_factor / 100. + .0001
            if not getattr(args, 'no_jpeg', False):
                encoded_image = extra_utils.jpeg_compress_decompress(encoded_image, rounding=extra_utils.round_only_at_0, factor=jpeg_factor)
            encoded_image = torch.clamp(encoded_image, 0, 1)
        else:
            p = torch.rand(1)[0]
            if p < 0.4:
                jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(getattr(args, 'jpeg_quality_ramp', 1)) * (100. - getattr(args, 'jpeg_quality', self.jpeg_quality))
                if jpeg_quality < 50:
                    jpeg_factor = 5000. / jpeg_quality
                else:
                    jpeg_factor = 200. - jpeg_quality * 2
                jpeg_factor = jpeg_factor / 100. + .0001
                if not getattr(args, 'no_jpeg', False):
                    encoded_image = extra_utils.jpeg_compress_decompress(encoded_image, rounding=extra_utils.round_only_at_0, factor=jpeg_factor)
                encoded_image = torch.clamp(encoded_image, 0, 1)
            elif p < 0.7:
                encoded_image = self.jpeg(encoded_image)
                encoded_image = torch.clamp(encoded_image, 0, 1)
            else:
                # MBRS path uses [-1,1] convention in your original code
                encoded_image_ = encoded_image * 2 - 1
                cover_img_ = cover_img * 2 - 1
                encoded_image_ = self.mbrs_noise([encoded_image_, cover_img_])
                encoded_image = (encoded_image_ + 1) * 0.5
                encoded_image = torch.clamp(encoded_image, 0, 1)

        # optional low-frequency projection (experimental; off by default)
        if self.enable_lowfreq_project and random.random() < 0.8:
            # make residual between encoded and cover, project the residual's lowfreq, and reconstruct
            if encoded_image.shape != cover_img.shape:
                encoded_image = F.interpolate(encoded_image, size=cover_img.shape[2:], mode='bilinear', align_corners=False)
            residual = encoded_image - cover_img
            residual_lf = self.maybe_lowfreq_project(residual, low_radius=12)
            encoded_image = cover_img + residual_lf
            encoded_image = torch.clamp(encoded_image, 0, 1)

        # convert back to [-1,1] to match original expectation
        encoded_image = encoded_image * 2 - 1
        encoded_image = encoded_image.to(encoded_image_type)
        return encoded_image


# Keep your existing helper classes with minimal edits (ImagenetCTransform, IG_Filter)
class ImagenetCTransform(nn.Module):
    def __init__(self, max_severity=5) -> None:
        super().__init__()
        self.max_severity = max_severity
        self.tform = RandomImagenetC(max_severity=max_severity, phase='train')

    def forward(self, x, corrupt_strength=None):
        original_shape = x.shape  # [B, C, H, W]
        img0 = x.detach().cpu().numpy()
        img = img0 * 127.5 + 127.5
        img = img.transpose(0, 2, 3, 1).astype(np.uint8)
        img = [Image.fromarray(i) for i in img]
        img = [self.tform(i, corrupt_strength=corrupt_strength) for i in img]
        img = np.array([np.array(i) for i in img], dtype=np.float32)
        img = img.transpose(0, 3, 1, 2) / 127.5 - 1.
        img = torch.from_numpy(img).to(x.device)
        
        # 确保尺寸与输入一致
        if img.shape != original_shape:
            img = F.interpolate(img, size=(original_shape[2], original_shape[3]), mode='bilinear', align_corners=False)
        
        return img


class IG_Filter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tform = [
            pilgram._1977,
            pilgram.aden,
            pilgram.brannan,
            pilgram.brooklyn,
            pilgram.clarendon,
            pilgram.earlybird,
            pilgram.gingham,
            pilgram.hudson,
            pilgram.inkwell,
            pilgram.kelvin,
            pilgram.lark,
            pilgram.lofi,
            pilgram.maven,
            pilgram.mayfair,
            pilgram.moon,
            pilgram.nashville,
            pilgram.perpetua,
            pilgram.reyes,
            pilgram.rise,
            pilgram.slumber,
            pilgram.stinson,
            pilgram.toaster,
            pilgram.valencia,
            pilgram.walden,
            pilgram.willow,
            pilgram.xpro2,
        ]
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def forward(self, x):
        processed_residual = []
        for single_img in x:
            encoded_pil_image = self.to_pil(single_img)
            selected_filter = random.choice(self.tform)
            filtered_img_pil = selected_filter(encoded_pil_image)
            filtered_img_tensor = self.to_tensor(filtered_img_pil).to(x.device)
            residual = filtered_img_tensor - single_img
            processed_residual.append(residual)
        processed_residual = torch.stack(processed_residual)
        x = x + processed_residual
        return x
