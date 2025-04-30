import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ViT 模型定义 (vit.py)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

from vit import ViT

# MLP 元数据模型定义
class MetadataMLP_PL(torch.nn.Module):
    def __init__(self, input_dim=27, hidden_dim=64, output_dim=2, dropout_p=0.1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish()
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.output_layer(x)

# 自定义数据集类
class ImageMetaLabelDataset(Dataset):
    def __init__(self, images_file, meta_file, label_file, image_size=224):
        with open(images_file, 'rb') as f:
            self.images = pickle.load(f)
        with open(meta_file, 'rb') as f:
            self.meta = pickle.load(f)
        with open(label_file, 'rb') as f:
            self.labels = pickle.load(f)

        assert len(self.images) == len(self.meta) == len(self.labels), "数据长度不一致"

        # 图像预处理管道
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        # 元数据转tensor
        self.meta = torch.tensor(np.array(self.meta), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 图像处理
        image = self.images[idx]
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # 处理图像维度
        if image.ndim == 1:
            image = image.reshape(224, 224, 3)
        elif image.shape[0] == 3 and image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))
        
        image = self.transform(image)
        return image, self.meta[idx], self.labels[idx]

# 模型加载函数
def load_vit_model(checkpoint_path, device='cuda'):
    """加载ViT图像模型"""
    config = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': 2,
        'dim': 768,
        'depth': 12,
        'heads': 12,
        'mlp_dim': 3072,
        'channels': 3,
        'dropout': 0.1,
        'emb_dropout': 0.1
    }
    
    vit = ViT(**config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('model.', '').replace('net.', '')
        new_state_dict[name] = v
    
    vit.load_state_dict(new_state_dict)
    vit.eval()
    return vit

def load_mlp_model(checkpoint_path, device='cuda'):
    """加载MLP元数据模型"""
    mlp = MetadataMLP_PL().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('model.', '').replace('net.', '')
        new_state_dict[name] = v
    
    mlp.load_state_dict(new_state_dict)
    mlp.eval()
    return mlp

# 获取全部预测结果
def get_all_predictions(model, dataloader, model_type='vit'):
    """获取模型对整个数据集的预测结果"""
    all_logits = []
    all_labels = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in dataloader:
            images, metas, labels = batch
            if model_type == 'vit':
                inputs = images.to(device)
            else:
                inputs = metas.to(device)
            
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    return (
        torch.cat(all_logits),
        torch.cat(all_labels)
    )

# 权重扫描函数
def evaluate_weighted_accuracy(vit_logits, mlp_logits, true_labels):
    """扫描不同权重组合，计算融合准确率"""
    results = []
    weights = np.arange(0.0, 1.01, 0.01)  # 从0到1，步长0.01
    
    for alpha in weights:
        # 加权融合 (alpha控制ViT贡献度)
        combined_logits = alpha * vit_logits + (1 - alpha) * mlp_logits
        
        # 计算准确率
        preds = torch.argmax(combined_logits, dim=1)
        accuracy = (preds == true_labels).float().mean().item()
        
        results.append({
            'alpha': round(alpha, 2),
            'accuracy': round(accuracy, 4)
        })
    
    return pd.DataFrame(results)

# 主程序
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型保存目录
    os.makedirs('results', exist_ok=True)
    
    # 加载数据集
    dataset = ImageMetaLabelDataset(
        images_file="dataset/images.pkl",
        meta_file="dataset/meta.pickle",
        label_file="dataset/labels.pkl"
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 加载模型
    print("🔄 加载ViT图像模型...")
    vit = load_vit_model("checkpoints/vit_best.ckpt", device=device)
    
    print("\n🔄 加载MLP元数据模型...")
    mlp = load_mlp_model("checkpoints/mlp_best.ckpt", device=device)
    
    # 获取所有预测结果
    print("\n🔍 获取ViT模型预测结果...")
    vit_logits, true_labels = get_all_predictions(vit, dataloader, model_type='vit')
    
    print("\n🔍 获取MLP模型预测结果...")
    mlp_logits, _ = get_all_predictions(mlp, dataloader, model_type='mlp')
    
    # 权重扫描
    print("\n🔍 开始扫描权重组合...")
    results_df = evaluate_weighted_accuracy(vit_logits, mlp_logits, true_labels)
    
    # 找到最佳权重
    best_row = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\n🏆 最佳权重组合: α={best_row['alpha']}, 准确率={best_row['accuracy']:.4f}")
    
    # 保存结果
    results_df.to_csv("results/weighted_results.csv", index=False)
    
    # 可视化
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='alpha', y='accuracy')
    plt.title('Ensemble Accuracy vs ViT Weight (α)')
    plt.xlabel('ViT Weight (α)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.savefig("results/weight_analysis.png")
    plt.show()