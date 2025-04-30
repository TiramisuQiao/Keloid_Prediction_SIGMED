# dataset.py

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

class ImageLabelDataset(Dataset):
    def __init__(self, images_file, label_file, image_size=224, channels=3):
        """
        Args:
            images_file: pkl 文件路径，包含图像数组列表
            label_file: pkl 文件路径，包含标签列表
            image_size: 输入图像尺寸
            channels: 图像通道数 (1=灰度图, 3=RGB)
        """
        # 加载数据
        with open(images_file, 'rb') as f:
            self.images = pickle.load(f)
        with open(label_file, 'rb') as f:
            self.labels = pickle.load(f)

        assert len(self.images) == len(self.labels), "数据长度不一致！"

        # 转换为 NumPy 数组（可选，视原始数据格式而定）
        if not isinstance(self.images[0], np.ndarray):
            self.images = [np.array(img) for img in self.images]

        # 图像预处理管道
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*channels, std=[0.5]*channels)
        ])

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx]

        # 确保是 NumPy 数组
        if not isinstance(image, np.ndarray):
            image = np.array(image)

    # 检查 shape（示例：假设图像是 224x224x3 的 RGB 图像）
        if image.ndim == 1:
        # 示例：如果是 224x224 的图像，将其 reshape 成 (224, 224, 3)
            image = image.reshape(224, 224, 3)

    # 如果是 CHW 格式，转成 HWC
        if image.shape[0] == 3 and image.ndim == 3:
            image = np.transpose(image, (1, 2, 0))

    # 应用 transform
        image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label
    
# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from vit import ViT

# 创建模型保存目录
os.makedirs('model_pth', exist_ok=True)

# 配置参数
config = {
    'images_file': "dataset/images.pkl",
    'label_file': "dataset/label.pickle",
    'image_size': 224,
    'patch_size': 16,
    'num_classes': 2,
    'dim': 768,
    'depth': 12,
    'heads': 12,
    'mlp_dim': 3072,
    'channels': 3,
    'batch_size': 16,
    'max_epochs': 100,
    'learning_rate': 1e-4,
    'train_val_split_ratio': 0.8,
    'save_best_acc': 0.9
}

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据集
dataset = ImageLabelDataset(
    config['images_file'],
    config['label_file'],
    image_size=config['image_size'],
    channels=config['channels']
)

# 数据划分
train_size = int(config['train_val_split_ratio'] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建 Dataloader
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# 初始化模型
model = ViT(
    image_size=config['image_size'],
    patch_size=config['patch_size'],
    num_classes=config['num_classes'],
    dim=config['dim'],
    depth=config['depth'],
    heads=config['heads'],
    mlp_dim=config['mlp_dim'],
    channels=config['channels']
).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# 训练循环
best_accuracy = 0.0

for epoch in range(config['max_epochs']):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train

    # 验证阶段
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = correct_val / total_val

    print(f"Epoch [{epoch+1}/{config['max_epochs']}], "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Acc: {val_acc:.4f}")

    scheduler.step(avg_train_loss)

    # 保存最佳模型
    if val_acc > best_accuracy and val_acc >= config['save_best_acc']:
        best_accuracy = val_acc
        save_path = os.path.join('model_pth', f'vit_best_{val_acc:.4f}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model to {save_path}")
        