import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import vine.src.Alias.training_src.extra_utils as extra_utils
import torch.nn.functional as F
from PIL import ImageFile
from datasets import load_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import json
import numpy as np
from glob import glob
from PIL import Image, ImageOps, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EditData(Dataset):
    def __init__(self, data_path="E:\phd//4\code\VINE\dataset//clip-filtered-dataset", secret_size=1, size=(512, 512)):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size


        self.t_256 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        self.to_tensor = transforms.ToTensor()

        # ======== 构建样本列表 ========
        self.samples = []
        folder_list = sorted(
            [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        )

        for folder in folder_list:
            prompt_path = os.path.join(folder, "prompt.json")
            if os.path.exists(prompt_path):
                try:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        prompt_data = json.load(f)
                        edit_text = prompt_data.get("edit", "")
                except json.JSONDecodeError:
                    print(f"[Warning] Invalid JSON in {prompt_path}")
                    edit_text = ""
            else:
                edit_text = ""

            # 找出所有 _0.jpg 图像
            img0_files = sorted(glob(os.path.join(folder, "*_0.jpg")))
            for img0_path in img0_files:
                # prefix = os.path.basename(img0_path).split("_0.jpg")[0]
                # img1_path = os.path.join(folder, f"{prefix}_1.jpg")
                # if os.path.exists(img1_path):
                    # 有完整的一对图像
                self.samples.append({
                        "img0": img0_path,
                        # "img1": img1_path,
                        "edit": edit_text
                    })
                # else:
                #     print(f"[Warning] Missing pair for {img0_path}")
        print('edit_text',img0_path,edit_text)
        print(f"Total pairs found: {len(self.samples)}")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):


        sample = self.samples[idx]
        img0_path = sample["img0"]
        edit_text = sample["edit"]

        img_cover = Image.open(img0_path).convert("RGB")
        img_cover_256 = self.t_256(img_cover)

        img_cover = ImageOps.fit(img_cover, self.size)
        img_cover = self.to_tensor(img_cover)

        # 归一化到 [-1, 1]
        img_cover = 2.0 * img_cover - 1.0
        img_cover_256 = 2.0 * img_cover_256 - 1.0

        # 生成1的概率为0.8，生成0的概率为0.2
        secret_flag = np.random.choice([0, 1], p=[0.4, 0.6])

        #         # 2. 根据标记生成全0或全1数组（长度=secret_size，整数类型）
        if secret_flag == 0:
            secret = np.zeros(self.secret_size, dtype=int)  # 全0数组
        else:
            secret = np.ones(self.secret_size, dtype=int)  # 全1数组

        secret = torch.from_numpy(secret).float()

        return {
            "cover_img": img_cover,
            "cover_img_256": img_cover_256,
            "secret": secret,
            "prompt": edit_text
        }


def get_secret_acc(secret_true, secret_pred):
    if 'cuda' in str(secret_pred.device):
        secret_pred = secret_pred.cpu()
        secret_true = secret_true.cpu()
    secret_pred = torch.round(secret_pred)
    correct_pred = torch.sum((secret_pred - secret_true) == 0, dim=1)
    str_acc = 1.0 - torch.sum((correct_pred - secret_pred.size()[1]) != 0).numpy() / correct_pred.size()[0]
    bit_acc = torch.sum(correct_pred).numpy() / secret_pred.numel()
    return bit_acc, str_acc


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params