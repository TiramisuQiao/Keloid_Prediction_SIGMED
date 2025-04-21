import torch
import pickle
import numpy as np

# 引入模型结构（必须保持和之前一致）
class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MetadataMLP(torch.nn.Module):
    def __init__(self, input_dim=27, hidden_dim=64, output_dim=2, dropout_p=0.1):
        super(MetadataMLP, self).__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            Swish(),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            Swish(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            Swish()
        )
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.output_layer(x)

# 加载模型
model = MetadataMLP(input_dim=27, hidden_dim=64, output_dim=2)
model.load_state_dict(torch.load("new_code/model.pth"))  # 加载参数
model.eval()  # 切换到推理模式

# 加载 scaler
with open("new_code/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 示例：预测一个新样本（必须是27维）
sample = [[0, 1, 0, 0, 4, 3, 3, 2, 0, 1, 0, 0, 1, 2, 5, 2, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0, 1]]
sample = scaler.transform(sample)  # 标准化
sample_tensor = torch.tensor(sample, dtype=torch.float32)

# 推理预测
with torch.no_grad():
    logit = model(sample_tensor)
    prob = torch.softmax(logit, dim=1)
    pred = torch.argmax(prob, dim=1).item()

print(f"预测结果：{pred}，概率分布：{prob.numpy()}")