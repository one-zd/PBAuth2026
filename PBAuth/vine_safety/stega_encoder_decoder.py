import torch, os
from torch import nn
import torch.nn.functional as F
from torchvision import models
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._validators import validate_repo_id
import matplotlib.pyplot as plt
import torch
import os


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
     
        
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.contiguous().view(input.size(0), -1)
    

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Decoder(nn.Module):
    def __init__(self, secret_size=100):
        super(Decoder, self).__init__()
        self.secret_size = secret_size
        # self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Conv2D(128, 128, 3, activation='relu'),
            Conv2D(128, 128, 3, strides=2, activation='relu'),
            Conv2D(128, 256, 3, strides=2, activation='relu'),
            Conv2D(256, 256, 3, activation='relu'),
            Conv2D(256, 256, 3, strides=2, activation='relu'),
            Conv2D(256, 512, 3, strides=2, activation='relu'),
            Conv2D(512, 512, 3, activation='relu'),
            Conv2D(512, 512, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(512, 256, activation='relu'),
            Dense(256, secret_size, activation=None))

    def forward(self, image):
        image = image - .5
        return torch.sigmoid(self.decoder(image))


def visualize_tensor(tensor, name):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import os

    os.makedirs("vis", exist_ok=True)

    t = tensor.detach().cpu()

    # 4D -> 3D
    if t.dim() == 4:
        t = t[0]

    # 3D -> C,H,W
    if t.dim() == 3:
        C, H, W = t.shape
        t_np = t.numpy()

        for i in range(C):
            ch = t_np[i]

            # --- 🔥 高级增强：分位数对比度拉伸 ---
            p1 = np.percentile(ch, 1)  # 1% 低值
            p99 = np.percentile(ch, 99)  # 99% 高值

            ch_norm = np.clip((ch - p1) / (p99 - p1 + 1e-8), 0, 1)

            plt.imshow(ch_norm, cmap='inferno')
            plt.colorbar()
            plt.savefig(f"vis/{name}_ch{i}.png")
            plt.close()

        print(f"[Saved enhanced {name}]")
        return

    # 2D Fallback
    if t.dim() == 2:
        t_np = t.numpy()
        p1 = np.percentile(t_np, 1)
        p99 = np.percentile(t_np, 99)
        t_norm = np.clip((t_np - p1) / (p99 - p1 + 1e-8), 0, 1)

        plt.imshow(t_norm, cmap='inferno')
        plt.colorbar()
        plt.savefig(f"vis/{name}.png")
        plt.close()
        return

    print(f"Cannot visualize tensor with shape {t.shape}")


class ConditionAdaptor(nn.Module):
    def __init__(self):
        super(ConditionAdaptor, self).__init__()
        
        self.secret_dense1 = Dense(100, 64 * 64, activation='relu')    ########100
        self.secret_dense2 = Dense(64 * 64, 3 * 64 * 64, activation='relu') 
        self.conv1 = Conv2D(6, 6, 3, activation='relu')
        self.conv2 = Conv2D(6, 3, 3, activation=None)
    
    # def forward(self, secrect, img_feature):
    #     B, C, H, W = img_feature.shape
    #     device = img_feature.device

    #     # print('secrect',secrect)
    #     # visualize_tensor(secrect, "secrect")
    #     # secret mask: (B,) boolean
    #     mask = (secrect.sum(dim=1) > 0)

    #     # 初始化 dense 输出为零
    #     s = torch.zeros(B, 3, 256, 256, device=device)

    #     if mask.any():
    #         s_valid = 2 * (secrect[mask, :].float() - 0.5)
    #         s_valid = self.secret_dense1(s_valid)
    #         s_valid = self.secret_dense2(s_valid)
    #         s_valid = s_valid.reshape(-1, 3, 64, 64)

    #         s_valid = nn.Upsample(scale_factor=(4, 4))(s_valid)
    #         s[mask] = s_valid
    #         # print('s_valid',s_valid.shape)  ######  (1,3,256,256)
    #         # visualize_tensor(s_valid, "s_valid")
    #     inputs = torch.cat([s, img_feature], dim=1)
    #     conv1 = self.conv1(inputs) 
    #     conv2 = self.conv2(conv1)

    #     # print('g1_conv2',conv2.shape)
    #     # visualize_tensor(conv2, "g1_conv2")

    #     return conv2

    def forward(self, secrect, img_feature):
        secrect = 2 * (secrect - 0.5)
        # print('secrect',secrect)
        # visualize_tensor(secrect, "secrect")
        secrect = self.secret_dense1(secrect)
        secrect = self.secret_dense2(secrect)
        secrect = secrect.reshape(-1, 3, 64, 64)
    
        secrect_enlarged = nn.Upsample(scale_factor=(4, 4))(secrect)
    
        # print('secrect_enlarged',secrect_enlarged.shape)  ######  (1,3,256,256)
        # visualize_tensor(secrect_enlarged, "secrect_enlarged")
    
        inputs = torch.cat([secrect_enlarged, img_feature], dim=1)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
    
        # print('conv2',conv2.shape)
        # visualize_tensor(conv2, "conv2")
    
        return conv2
    
    
class ConditionAdaptor_orig(nn.Module):
    def __init__(self):
        super(ConditionAdaptor_orig, self).__init__()
        
        self.secret_dense1 = Dense(100, 64 * 64, activation='relu') 
        self.secret_dense2 = Dense(64 * 64, 3 * 64 * 64, activation='relu') 

        self.conv1 = Conv2D(6, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(70, 32, 3, activation='relu')
        self.conv10 = Conv2D(32,32,3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)

    def forward(self, secrect, image):
        secrect = secrect - .5   

        secrect = self.secret_dense1(secrect)  
        secrect = self.secret_dense2(secrect)  
        secrect = secrect.reshape(-1, 3, 64, 64) 
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(secrect) 

        inputs = torch.cat([secrect_enlarged, image], dim=1)  
        conv1 = self.conv1(inputs) 
        conv2 = self.conv2(conv1)  
        conv3 = self.conv3(conv2)  
        conv4 = self.conv4(conv3)  
        conv5 = self.conv5(conv4)  
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5)) 
        merge6 = torch.cat([conv4, up6], dim=1)  
        conv6 = self.conv6(merge6)  
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6))  
        merge7 = torch.cat([conv3, up7], dim=1) 
        conv7 = self.conv7(merge7) 
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7)) 
        merge8 = torch.cat([conv2, up8], dim=1) 
        conv8 = self.conv8(merge8)  
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8))  
        merge9 = torch.cat([conv1, up9, inputs], dim=1)  
        
        conv9 = self.conv9(merge9) 
        conv10=self.conv10(conv9) 
        residual = self.residual(conv10)
        return residual
    
    
class CustomConvNeXt(nn.Module, PyTorchModelHubMixin):
    def __init__(self, secret_size=100, ckpt_path=None, device=None, config=None):
        super(CustomConvNeXt, self).__init__()
        self.convnext = models.convnext_base()
        self.convnext.classifier.append(nn.Linear(in_features=1000, out_features=secret_size, bias=True))
        self.convnext.classifier.append(nn.Sigmoid())
    
        if ckpt_path is not None:
            self.load_ckpt_from_state_dict(ckpt_path, device)
            
    def load_ckpt_from_state_dict(self, ckpt_path, device):
        self.convnext.load_state_dict(torch.load(os.path.join(ckpt_path, 'CustomConvNeXt.pth')))
        self.convnext.to(device)

    def forward(self, x):
        x = self.convnext(x)
        return x

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