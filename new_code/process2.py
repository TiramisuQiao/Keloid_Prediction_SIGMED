#机器学习
import pickle  # 用于加载和保存数据
import numpy as np  # 处理数据的常用库
from sklearn.model_selection import train_test_split  # 用于将数据拆分为训练集和测试集
from sklearn.preprocessing import StandardScaler  # 对数据进行标准化，让特征值在同一个尺度上
from sklearn.ensemble import RandomForestClassifier  # 一个机器学习分类模型，适用于处理分类任务
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型，用于二分类问题的预测
from sklearn.metrics import accuracy_score  # 计算模型的预测准确率

#    1.加载数据
with open('new_code/data_1.pickle', 'rb') as f:
    X = pickle.load(f)#X 是包含特征数据的数组，每一行代表一个样本，27个特征。
with open('new_code/data_2.pickle', 'rb') as f:
    y = pickle.load(f)#y 是标签数据（目标值），通常是你要预测的类别（0 或 1）。

#   2.数据预处理
X = np.array(X)# 将列表转换为 numpy 数组
y = np.array(y)
scaler = StandardScaler()# 数据标准化
X = scaler.fit_transform(X)

# 3.划分训练/测试集,拆分数据集，这样我们就能用训练集训练模型，用测试集验证模型的效果。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# train_test_split：将数据拆分为训练集（80%）和测试集（20%）# random_state=42 是为了确保每次拆分结果相同。

print("X_train.shape",X_train.shape)
print("X_test.shape",X_test.shape)
# 4.创建模型，训练
# model = RandomForestClassifier(n_estimators=100, random_state=42)# 随机森林模型
"""
RandomForestClassifier是 Scikit-Learn（sklearn）库中提供的一种 集成机器学习模型，专门用来处理分类问题，比如二分类（0 或 1）或多分类任务。它的核心思想是：构建很多颗决策树（tree），每棵树都独立做预测，最后用“投票”来决定最终预测结果。
n_estimatohrs：表示要构建 100 棵决策树，越多的树 → 模型越稳定，准确率越高（但训练时间变长），通常设置为 100、200、500 等等都可以，100 是一个合理的默认值。
random_state：设置随机种子，保证每次运行程序时，模型的结果是一样的。因为随机森林在建树过程中有“随机性”，每次不设置种子就可能结果不同。42 是一个常用的“神秘数字”，其实你写 123、2024、0 也可以，只要你以后想复现实验，值不变就行。
"""
model = LogisticRegression() #逻辑回归模型
"""
LogisticRegression() 是 逻辑回归模型 的初始化方法，用来创建一个逻辑回归分类器的实例。，逻辑回归其实不是“回归”，而是一个 用于二分类问题 的模型。
它有一个数学形式
x 是输入特征（你的 27 个特征） ，w 是权重（模型学出来的），b 是偏置项最终输出是一个 0~1 的概率，再根据 >0.5 / <0.5 决定是 1 还是 0
"""

model.fit(X_train, y_train)# 训练模型

# 5.验证,评估模型
y_pred = model.predict(X_test)# 用测试集预测结果
acc = accuracy_score(y_test, y_pred)# 计算准确率
print("Test accuracy:", acc)

# 6.对新的单样本进行预测
sample = np.random.randint(0, 6, size=(1, 27))  # 随机模拟一个新样本,（27个特征值）
sample_norm = scaler.transform(sample)# 对新样本进行标准化,与训练数据保持一致
prediction = model.predict(sample_norm)# 使用训练好的模型来对新样本进行预测，返回预测的类别（0 或 1）
print("Sample prediction:", prediction[0])