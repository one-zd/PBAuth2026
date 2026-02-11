import os
import sys, gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from huggingface_hub import PyTorchModelHubMixin
from vine.src.Alias.vine_safety.stega_encoder_decoder import ConditionAdaptor
from vine.src.model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._validators import validate_repo_id

class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.mode() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded


def initialize_unet(rank, return_lora_module_names=False):
    unet = UNet2DConditionModel.from_pretrained("E:\phd//4\code\VINE\stabilityaisd_turbo", subfolder="unet",from_safetensors=True)
    unet.requires_grad_(False)
    unet.train()
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder, lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder, lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others, lora_alpha=rank)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


def initialize_unet_no_lora(path="E:\phd//4\code\VINE\stabilityaisd_turbo"):
    unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet",from_safetensors=True)
    unet.requires_grad_(True)
    unet.train()
    return unet
    

def initialize_vae(rank=4, return_lora_module_names=False):
    vae = AutoencoderKL.from_pretrained("E:\phd//4\code\VINE\stabilityaisd_turbo", subfolder="vae",from_safetensors=True)
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1
    l_vae_target_modules = ["conv1","conv2","conv_in", "conv_shortcut",
        "conv", "conv_out", "skip_conv_1", "skip_conv_2", "skip_conv_3", 
        "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0",
    ]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae
    
    
def initialize_vae_no_lora(path="E:\phd//4\code\VINE\stabilityaisd_turbo"):
    vae = AutoencoderKL.from_pretrained(path, subfolder="vae",from_safetensors=True)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()
    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True).cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1

    return vae


class VINE_Turbo(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, ckpt_path=None, device='cuda', config=None):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained("E:\phd//4\code\VINE\stabilityaisd_turbo", subfolder="tokenizer", use_fast=False,from_safetensors=True)
        text_encoder = CLIPTextModel.from_pretrained("E:\phd//4\code\VINE\stabilityaisd_turbo", subfolder="text_encoder")
        text_encoder.requires_grad_(False)
        text_encoder.to(device)

        fixed_a2b_tokens = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
        self.fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.unsqueeze(0).to(device))[0].detach()

        del text_encoder, tokenizer, fixed_a2b_tokens # free up some memory
        gc.collect()
        torch.cuda.empty_cache()

        self.sec_encoder = ConditionAdaptor()
        self.unet = initialize_unet_no_lora()
        self.vae_a2b = initialize_vae_no_lora()
        self.vae_enc = VAE_encode(self.vae_a2b)
        self.vae_dec = VAE_decode(self.vae_a2b)
        self.sched = make_1step_sched(device)
        self.timesteps = torch.tensor([self.sched.config.num_train_timesteps - 1] * 1, device=device).long()
        
        if ckpt_path is not None:
            self.load_ckpt_from_state_dict(ckpt_path, device)

        # print(f"VAE 权重加载自：E:\phd//4\code\VINE\stabilityaisd_turbo")
        # print(f"VAE 示例权重：{self.vae_a2b.encoder.conv_in.weight.shape}")  # 应输出合法形状（如 [320, 3, 3, 3]）
            
    def load_ckpt_from_state_dict(self, ckpt_path, device):
        # self.sec_encoder.load_state_dict(torch.load(os.path.join(ckpt_path, 'ConditionAdaptor.pth')), strict=False)

        ckpt = torch.load(os.path.join(ckpt_path, 'ConditionAdaptor.pth'))
        model_dict = self.sec_encoder.state_dict()
        # 过滤掉形状不匹配的权重
        filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_ckpt)
        self.sec_encoder.load_state_dict(model_dict)
        self.sec_encoder.to(device)
        # self.sec_encoder.eval()

        self.unet.load_state_dict(torch.load(os.path.join(ckpt_path, 'UNet2DConditionModel.pth')))
        self.unet.to(device)
        # self.unet.requires_grad_(False)
        # self.unet.eval()
        
        self.vae_a2b.load_state_dict(torch.load(os.path.join(ckpt_path, 'vae.pth')))
        self.vae_a2b.to(device)
        # self.vae_a2b.requires_grad_(False)
        # self.vae_a2b.eval()

    @staticmethod
    def get_traininable_params(unet=None, vae_a2b=None, vae_b2a=None):
        # add all unet parameters
        params_gen = []
        if unet is not None:
            params_gen = params_gen + list(unet.conv_in.parameters())
            unet.conv_in.requires_grad_(True)
            unet.set_adapters(["default_encoder", "default_decoder", "default_others"])
            for n,p in unet.named_parameters():
                # if "lora" in n and "default" in n:
                #     assert p.requires_grad
                if p.requires_grad:
                    params_gen.append(p)
        
        # add all vae_a2b parameters
        if vae_a2b is not None:
            for n,p in vae_a2b.named_parameters():
                # if "lora" in n and "vae_skip" in n:
                #     assert p.requires_grad
                if p.requires_grad:
                    params_gen.append(p)
            params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
            params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
            params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
            params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        if vae_b2a is not None:
            for n,p in vae_b2a.named_parameters():
                if "lora" in n and "vae_skip" in n:
                    assert p.requires_grad
                    params_gen.append(p)
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
            params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())
            
        return params_gen

    @classmethod
    def _from_pretrained(
            cls,
            model_id: str,
            revision: str = None,
            cache_dir: str = None,
            force_download: bool = False,
            proxies: dict = None,
            resume_download: bool = False,
            local_files_only: bool = False,
            token: str = None, **kwargs,
    ):
        # 1. 提取模型初始化参数（如 device、config）
        model_kwargs = kwargs.pop("model_kwargs", {})
        # 2. 定义你的 safetensors 文件名（必须与本地文件一致！）
        model_filename = "model.safetensors"  # 若你的文件是其他名字，比如"VINE-B-Enc.safetensors"，则修改这里
        # 3. 区分“本地路径”和“远程仓库”，获取模型文件路径
        try:
            # 尝试验证是否为远程Hugging Face仓库ID（如 "username/VINE-B-Enc"）
            validate_repo_id(model_id)
            # 远程仓库：从Hugging Face Hub下载safetensors文件
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=model_filename,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        except:
            # 本地路径：直接拼接本地文件路径
            model_file = os.path.join(model_id, model_filename)
            # 检查本地文件是否存在，避免后续报错
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"本地模型文件不存在：{model_file}\n请确认文件名是否为 {model_filename}")

        # 4. 确定权重加载设备（优先用 model_kwargs 中的 device，默认 CPU）
        map_location = model_kwargs.get("device", "cpu")
        if isinstance(map_location, str):
            map_location = torch.device(map_location)

        # 5. 加载 safetensors 权重（必须用 safetensors.torch.load_file，不能用 torch.load）
        state_dict = load_file(model_file, device=str(map_location))

        # 6. 实例化模型并加载权重
        model = cls(**model_kwargs)
        model.load_state_dict(state_dict, strict=False)

        return model

    def forward(self, x, timesteps=None, secret=None):
        if timesteps == None:
            timesteps = self.timesteps
        B = x.shape[0]

        # print("22222222222secret",secret.shape)
        # print("33333333x",x.shape)

        x_sec = self.sec_encoder(secret, x)
        x_enc = self.vae_enc(x_sec, direction="a2b").to(x.dtype)

        BB = x_enc.shape[0]

        encoder_hidden_states = self.fixed_a2b_emb_base

        # 如果只有一个 prompt embedding，就直接 expand 到 batch
        if encoder_hidden_states.shape[0] == 1:
            encoder_hidden_states = encoder_hidden_states.expand(BB, -1, -1)
        # 如果大小对不上，就强制 repeat 再切
        elif encoder_hidden_states.shape[0] != BB:
            repeats = (BB + encoder_hidden_states.shape[0] - 1) // encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.repeat(repeats, 1, 1)[:BB]



        model_pred = self.unet(x_enc, timesteps, encoder_hidden_states=encoder_hidden_states,).sample.to(x.dtype)
        x_out = torch.stack([self.sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = self.vae_dec(x_out, direction="a2b").to(x.dtype)
        return x_out_decoded