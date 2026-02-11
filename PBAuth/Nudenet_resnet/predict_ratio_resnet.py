import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import resnet_our
from tqdm import tqdm

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        # 支持的图片扩展名
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
        
        # 递归遍历目录
        for root, _, files in os.walk(root_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(root, file))
        
        if not self.image_paths:
            print(f"Warning: No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空的 tensor 标记错误，或者处理方式由 collate_fn 决定
            # 这里简单起见，如果出错可能导致 DataLoader 报错，
            # 实际生产中应该用自定义 collate_fn 过滤 None
            return torch.zeros(3, 224, 224), img_path

def build_model(num_classes, device_ids):
    # 使用 resnet_our.resnet50，与训练一致
    model = resnet_our.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
    return model

def main(args):
    # 1. 准备数据预处理 (与训练保持一致)
    normalize = transforms.Normalize(mean=[0.5104833, 0.45094156, 0.35637376],
                                     std=[0.17324965, 0.15483698, 0.14012936])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # 2. 加载数据集
    print(f"Scanning images in {args.data_dir} ...")
    dataset = UnlabeledImageDataset(args.data_dir, transform=transform)
    if len(dataset) == 0:
        print("No images found. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, pin_memory=True)

    # 3. 加载 Checkpoint 和 模型
    print(f"Loading checkpoint from {args.checkpoint} ...")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    classes = checkpoint.get('classes')
    if not classes:
        raise ValueError("Checkpoint does not contain 'classes' info.")
    
    print(f"Model classes: {classes}")
    
    # 确定目标类别的索引
    # 默认寻找包含 'porn' 或 'nude' 的类别，不区分大小写
    target_class_indices = []
    target_keywords = ['porn', 'nude']
    
    for idx, class_name in enumerate(classes):
        # 检查类名是否包含关键词
        if any(keyword in class_name.lower() for keyword in target_keywords):
            target_class_indices.append(idx)
    
    if not target_class_indices:
        print(f"Warning: Could not automatically find a class matching keywords {target_keywords}.")
        print("Please manually verify the class names above.")
        # 如果找不到，默认取 index 0 (根据之前的代码 porn 0)
        print("Defaulting to index 0 as target class.")
        target_class_indices = [0]
    else:
        target_names = [classes[i] for i in target_class_indices]
        print(f"Target classes for 'nude' ratio: {target_names} (Indices: {target_class_indices})")

    model = build_model(len(classes), args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 4. 推理
    print("Starting inference ...")
    total_images = 0
    nude_images = 0
    
    # 用于保存结果详情（可选）
    results = []

    with torch.no_grad():
        for images, paths in tqdm(dataloader):
            images = images.cuda()
            
            # resnet_our 返回 (logits, pool)
            outputs, _ = model(images)
            
            # 获取预测类别
            _, preds = torch.max(outputs, 1)
            
            preds_cpu = preds.cpu().numpy()
            
            batch_size = len(preds_cpu)
            total_images += batch_size
            
            for i in range(batch_size):
                pred_idx = preds_cpu[i]
                is_nude = pred_idx in target_class_indices
                if is_nude:
                    nude_images += 1
                
                # 如果需要，可以保存每张图的预测结果
                # results.append((paths[i], classes[pred_idx]))

    # 5. 输出结果
    if total_images > 0:
        ratio = nude_images / total_images
        print("\n" + "="*30)
        print(f"Total Images: {total_images}")
        print(f"Nude/Porn Predicted: {nude_images}")
        print(f"Ratio: {ratio:.4%}")
        print("="*30 + "\n")
        
        # 可选：保存结果到文件
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(f"Total Images: {total_images}\n")
                f.write(f"Nude Count: {nude_images}\n")
                f.write(f"Ratio: {ratio:.4f}\n")
            print(f"Results saved to {args.output_file}")
            
    else:
        print("No images processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Nude/Porn Ratio on Unlabeled Dataset')
    
    parser.add_argument('--data-dir', default='F:\data_phd//4\W_bench_en\PGD_ep32_al2_st200_hsv1', help='Path to the folder containing images')
    parser.add_argument('--checkpoint',default='E:\phd//1pppd\code//NudeNet//nsfw-resnet-master//checkpoints_nudenet_resnet50//model_138_50.pth', help='Path to the model checkpoint')
    parser.add_argument('--device', default=[0], type=int, nargs='+', help='Device IDs (e.g. 0 1)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('-j', '--workers', default=8, type=int, help='Number of workers')
    parser.add_argument('--output-file', default=None, help='Path to save the result text file')

    args = parser.parse_args()
    
    main(args)
