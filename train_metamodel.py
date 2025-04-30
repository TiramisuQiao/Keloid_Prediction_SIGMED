import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pytorch_lightning as pl
import torchmetrics
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import LabelEncoder

os.makedirs('model_pth', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MetadataMLP_PL(pl.LightningModule):
    def __init__(self, input_dim=27, hidden_dim=64, output_dim=2, dropout_p=0.1, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish()
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 损失函数和指标
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.output_layer(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 记录指标
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log('train/loss', loss, prog_bar=False)
        self.log('train/acc', self.train_accuracy, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', self.val_accuracy, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)  # Forward pass
        loss = self.criterion(logits, labels)  # Compute loss

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)  # 可以复用 validation 的指标

        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.val_accuracy, prog_bar=True)

        return loss

class MetadataDataset(Dataset):
    def __init__(self, meta_file, label_file):
        """
        初始化数据集
        :param images_file: 图像文件路径 (.pkl)
        :param meta_file: 元数据文件路径 (.pkl)
        :param label_file: 标签文件路径 (.pkl)
        """
        # 加载数据
        with open(meta_file, 'rb') as f:
            self.meta = pickle.load(f)
        with open(label_file, 'rb') as f:
            self.labels = pickle.load(f)

        # 转换为 NumPy 数组
        self.meta = np.array(self.meta)
        self.labels = np.array(self.labels)


        assert  len(self.meta) == len(self.labels), "数据长度不一致！"

        # 自动编码字符串列
        if self.meta.dtype == object:
            print("🔍 检测到元数据包含非数值类型，正在自动编码...")
            self._encode_meta()
        else:
            print("✔️ 元数据已为数值类型，跳过编码")

        self.meta = self.meta.astype(np.float32)
        print(self.meta.shape)
        self.labels = self.labels.astype(np.int64)

    def _encode_meta(self):
        """
        对每一列进行判断，若含字符串则使用 LabelEncoder 编码
        """
        encoded_columns = []
        for col_idx in range(self.meta.shape[1]):
            column = self.meta[:, col_idx]
            if any(isinstance(x, str) for x in column):
                le = LabelEncoder()
                encoded_col = le.fit_transform(column).astype(float)
                encoded_columns.append(encoded_col)
            else:
                encoded_columns.append(column.astype(float))
        self.meta = np.column_stack(encoded_columns)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        返回单个样本 (image, meta, label)，均转换为 Tensor
        """
        meta = self.meta[idx]
        label = self.labels[idx]

        # 转换为 PyTorch 张量
        meta = torch.tensor(meta, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return meta, label
    
# --- 4. 主程序入口 ---
if __name__ == "__main__":
    # 配置参数
    config = {
        'meta_file': "dataset/meta.pickle",
        'label_file': "dataset/label.pickle",
        'input_dim': 27,
        'hidden_dim': 64,
        'output_dim': 2,
        'dropout_p': 0.1,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'max_epochs': 200,
        'train_val_split': 0.8,
        'patience': 15
    }
    
    print("🔧 正在初始化训练流程...")
    
    # 加载数据集
    dataset = MetadataDataset(config['meta_file'], config['label_file'])
    if len(dataset) == 0:
        raise RuntimeError("数据集加载失败，请检查数据文件")

    # 数据集划分
    train_size = int(config['train_val_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'],
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # 初始化模型
    model = MetadataMLP_PL()
    
    # 回调函数
    checkpoint = ModelCheckpoint(
        dirpath='model_pth',
        filename='best_model-{epoch:02d}-{val/acc:.4f}',
        monitor='val/acc',
        mode='max',
        save_top_k=1,
        save_last=True
    )
    
    early_stop = EarlyStopping(
        monitor='val/loss',
        min_delta=0.001,
        patience=config['patience'],
        verbose=True,
        mode='min'
    )
    
    logger = TensorBoardLogger("logs", name="metadata_classifier")
    
    # 训练器配置
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else 'auto',
        max_epochs=config['max_epochs'],
        callbacks=[checkpoint, early_stop],
        logger=logger,
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32
    )
    
    print("🚀 开始训练...")
    trainer.fit(model, train_loader, val_loader)
    
    print("📊 最终模型评估:")
    result = trainer.test(model, dataloaders=val_loader)
    print(f"最终结果: {result}")
    
    print("🧠 模型已保存至: model_pth/")