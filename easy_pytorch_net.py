import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

# 生成简单的数字数据集
# 任务：根据输入的数字特征预测数字类别（0-9）
def generate_simple_dataset(num_samples=1000):
    """
    生成简单的数字分类数据集（无需外部下载）
    
    数据集设计原理：
    - 输入特征：10维向量，采用"独热编码"思想，第i位为1表示数字i
    - 输出标签：对应的数字类别（0-9）
    - 添加高斯噪声模拟真实数据的不确定性
    
    Args:
        num_samples (int): 生成的样本数量，默认1000个
        
    Returns:
        tuple: (X, y)
            - X: torch.Tensor, 形状为(num_samples, 10)，特征矩阵
            - y: torch.Tensor, 形状为(num_samples,)，标签数组（0-9）
    """
    # 初始化特征矩阵 X，形状为(样本数, 特征维度=10)
    # 每行代表一个样本，每列代表一个特征维度
    X = np.zeros((num_samples, 10))
    
    # 初始化标签数组 y，存储每个样本对应的数字类别
    # dtype=np.int64 确保标签为整数类型，符合PyTorch分类任务要求
    y = np.zeros(num_samples, dtype=np.int64)
    
    # 逐个生成样本
    for i in range(num_samples):
        # 随机选择一个数字（0-9）作为该样本的真实标签
        digit = np.random.randint(0, 10)
        y[i] = digit  # 记录标签
        
        # 设置主特征：将对应数字位置的特征设为1
        # 例如：digit=3时，X[i, 3] = 1.0
        X[i, digit] = 1.0
        
        # 添加高斯噪声（均值=0，标准差=0.1）
        # 目的：模拟真实数据中的干扰，使任务更具挑战性
        # 噪声幅度较小（0.1），不会掩盖主特征的显著性
        X[i] += np.random.normal(0, 0.1, 10)
    
    # 将NumPy数组转换为PyTorch张量
    # X: float32类型（神经网络输入常用类型）
    # y: long类型（分类任务标签常用类型）
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# 定义简单的神经网络模型
class SimpleNN(nn.Module):
    """
    简单的两层全连接神经网络（用于数字分类任务）
    
    网络结构：输入层 -> 隐藏层(ReLU) -> 输出层
    
    Args:
        input_size (int): 输入特征维度，默认10（对应10个数字特征）
        hidden_size (int): 隐藏层神经元数量，默认20
        num_classes (int): 输出类别数，默认10（对应数字0-9）
    """
    def __init__(self, input_size=10, hidden_size=20, num_classes=10):
        # 调用父类 nn.Module 的构造函数
        super(SimpleNN, self).__init__()
        
        # 定义网络层
        # 第一层：全连接层，将输入特征映射到隐藏层
        # 参数: in_features=input_size, out_features=hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层 -> 隐藏层
        
        # ReLU激活函数：引入非线性，使网络能学习复杂模式
        # ReLU(x) = max(0, x)，解决梯度消失问题
        self.relu = nn.ReLU()
        
        # 第二层：全连接层，将隐藏层输出映射到类别概率
        # 参数: in_features=hidden_size, out_features=num_classes
        self.fc2 = nn.Linear(hidden_size, num_classes) # 隐藏层 -> 输出层

    def forward(self, x):
        """
        定义神经网络的前向传播路径
        
        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, input_size)
            
        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, num_classes)
                         每一行代表一个样本在各个类别上的得分（未经过softmax）
        """
        # 第一步：通过第一层全连接层
        out = self.fc1(x)  # 形状: (batch_size, hidden_size)
        
        # 第二步：应用ReLU激活函数
        out = self.relu(out)  # 形状: (batch_size, hidden_size)
        
        # 第三步：通过第二层全连接层，得到最终输出
        out = self.fc2(out)  # 形状: (batch_size, num_classes)
        
        return out

# 设置超参数
batch_size = 32
learning_rate = 0.01
num_epochs = 20

# 生成训练和测试数据
print("生成数据集...")
train_X, train_y = generate_simple_dataset(num_samples=1000)
test_X, test_y = generate_simple_dataset(num_samples=200)

# 创建数据集和数据加载器
# TensorDataset将特征和标签打包成数据集对象
train_dataset = TensorDataset(train_X, train_y)  # 训练数据集
test_dataset = TensorDataset(test_X, test_y)    # 测试数据集

# DataLoader创建数据迭代器，支持批量加载和打乱
# shuffle=True: 训练时打乱数据顺序，防止模型学习到数据顺序特征
# shuffle=False: 测试时不需要打乱，保持数据顺序便于评估
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失（用于分类任务）
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
print("\n开始训练...")

# 初始化训练历史记录列表，用于后续绘制训练曲线
train_loss_history = []  # 记录每个epoch的平均损失
train_acc_history = []   # 记录每个epoch的训练准确率

# 外层循环：遍历训练轮数（epoch）
for epoch in range(num_epochs):
    # 将模型设置为训练模式（启用dropout、batch normalization等训练相关操作）
    model.train()
    
    # 初始化本轮训练的累计损失和正确预测数
    running_loss = 0.0  # 累计损失
    correct = 0         # 正确预测样本数
    total = 0           # 总样本数
    
    # 内层循环：遍历训练数据集的每个批次（batch）
    for inputs, labels in train_loader:
        # 将数据移动到指定设备（GPU或CPU）
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # ========== 前向传播 ==========
        # 通过模型获取预测输出
        outputs = model(inputs)
        # 计算预测值与真实标签之间的损失
        loss = criterion(outputs, labels)
        
        # ========== 反向传播和优化 ==========
        # 清空上一轮的梯度，防止梯度累积
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 根据梯度更新模型参数
        optimizer.step()
        
        # ========== 统计本轮训练数据 ==========
        # 累加当前批次的损失值（item()将张量转换为标量）
        running_loss += loss.item()
        # 获取预测结果（取输出中概率最大的类别）
        _, predicted = torch.max(outputs.data, 1)
        # 累加总样本数
        total += labels.size(0)
        # 累加正确预测数
        correct += (predicted == labels).sum().item()
    
    # ========== 计算本轮训练指标 ==========
    avg_loss = running_loss / len(train_loader)  # 平均损失
    train_acc = 100 * correct / total           # 训练准确率（百分比）
    
    # 记录本轮指标到历史列表
    train_loss_history.append(avg_loss)
    train_acc_history.append(train_acc)
    
    # 打印本轮训练结果
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%')

# ========== 测试模型 ==========
print("\n开始测试...")

# 将模型设置为评估模式（禁用dropout、batch normalization等训练相关操作）
model.eval()

# 使用 torch.no_grad() 上下文管理器，禁用梯度计算以节省内存和加速推理
with torch.no_grad():
    correct = 0  # 正确预测数
    total = 0    # 总样本数
    
    # 遍历测试数据集的每个批次
    for inputs, labels in test_loader:
        # 将数据移动到指定设备
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向传播获取预测输出
        outputs = model(inputs)
        # 获取预测类别（取输出中概率最大的索引）
        _, predicted = torch.max(outputs.data, 1)
        
        # 累加统计数据
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算并打印测试准确率
    print(f'测试准确率: {100 * correct / total:.2f}%')

# ========== 展示预测结果示例 ==========
print("\n展示预测结果示例:")
model.eval()
with torch.no_grad():
    # 获取测试数据加载器的第一批数据
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    
    # 前向传播获取预测结果
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    
    # 打印表头
    print("\n样本 | 预测 | 真实")
    print("------|------|------")
    
    # 显示前10个样本的预测结果与真实标签对比
    for i in range(min(10, len(labels))):
        print(f"  {i+1:2d}  |  {predicted[i].item():2d}  |  {labels[i].item():2d}")

# 绘制训练结果图
print("\n绘制训练结果图...")
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_loss_history, 'b-', linewidth=2, label='训练损失')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('训练损失变化曲线', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_acc_history, 'r-', linewidth=2, label='训练准确率')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('准确率 (%)', fontsize=12)
plt.title('训练准确率变化曲线', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.ylim([0, 105])  # 设置y轴范围

plt.tight_layout()  # 调整子图间距
plt.show()

print("\n运行完成!")