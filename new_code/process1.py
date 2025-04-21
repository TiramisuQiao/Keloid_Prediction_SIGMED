# 深度学习实现数据预测分析
import numpy as np
import torch    #张量操作及深度学习模型的构建功能。
import torch.nn as nn   #神经网络模块，提供了构建神经网络所需的层和工具。
import torch.nn.functional as F #常见的激活函数和损失函数等操作
# 自定义的激活函数，比传统的 ReLU 有时效果更好
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 自定义神经网络模型类，主要用来处理元数据（metadata）
# 也就是非图像的结构化数据，比如：数值、类别、统计量等。
class MetadataMLP(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=64, output_dim=2, dropout_p=0.1):

        super(MetadataMLP, self).__init__()

        # 这是一个 3 层全连接神经网络
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
        # 这是最终输出的线性层，把隐藏层的输出变成你想要的输出维度。默认输出 2 个值，对应于二分类中的每个类别的得分
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    

# 张量经过特征提取器处理
    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        x = self.feature_extractor(x)
        out = self.output_layer(x)
        
        return out

if __name__ == "__main__":
    # 加载数据
    with open('new_code/data_1.pickle', 'rb') as f:
        X = pickle.load(f)
    with open('new_code/data_2.pickle', 'rb') as f:
        y = pickle.load(f)
    X = np.array(X)
    y = np.array(y)
    input_dim = X.shape[1]  # << 定义 input_dim

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 5. 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 6. 转张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8)

    # 7. 模型、损失与优化器
    model = MetadataMLP(input_dim=input_dim, hidden_dim=64, output_dim=2, dropout_p=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 8. 训练
    for epoch in range(20):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # 9. 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            preds = torch.argmax(output, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"Validation Accuracy: {correct/total:.2%}")

    # 10. 对单条新样本进行预测
    sample = np.random.randint(0, 6, size=(1, input_dim))
    sample_norm = scaler.transform(sample)
    sample_tensor = torch.tensor(sample_norm, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        logit = model(sample_tensor)
        pred = torch.argmax(logit, dim=1).item()
    print("Sample prediction:", pred)

# 保存模型参数到文件
torch.save(model.state_dict(), "new_code/model.pth")

# 保存标准化器（scaler）对象，用于预测时做相同的数据归一化
with open("new_code/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("模型和Scaler已保存")











    # batch_size = 8
    # input_dim = 32
    # model = MetadataMLP(input_dim=input_dim, hidden_dim=64, output_dim=2, dropout_p=0.1)
    # sample_input = torch.randn(batch_size, input_dim)
    # output = model(sample_input)
    # print("Test Metadata output:", output.shape)  
