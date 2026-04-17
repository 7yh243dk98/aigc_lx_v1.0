"""第一个神经网络完整示例 - 从零到一
目标：训练一个神经网络来解决简单的二分类问题（学习AND逻辑门）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

print("=" * 50)
print("第一步：准备训练数据")
print("=" * 50)

# 创建一个简单的数据集：AND逻辑门
# 输入：[0,0], [0,1], [1,0], [1,1]
# 输出：  0,     0,     0,     1
X = torch.tensor([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=torch.float32)

y = torch.tensor([[0],
                  [0],
                  [0],
                  [1]], dtype=torch.float32)

print(f"输入数据 X:\n{X}")
print(f"\n标签 y:\n{y}")


print("\n" + "=" * 50)
print("第二步：定义神经网络模型")
print("=" * 50)

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    """一个简单的两层神经网络
    
    结构：
    输入层（2个神经元）-> 隐藏层（4个神经元）-> 输出层（1个神经元）
    """
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 第一层：从2个输入到4个隐藏神经元
        self.fc1 = nn.Linear(2, 4)
        # 第二层：从4个隐藏神经元到1个输出
        self.fc2 = nn.Linear(4, 1)
    
    def forward(self, x):
        """前向传播：数据如何在网络中流动"""
        # x通过第一层，然后应用ReLU激活函数
        x = torch.relu(self.fc1(x))
        # x通过第二层，然后应用Sigmoid激活函数（输出0-1之间）
        x = torch.sigmoid(self.fc2(x))
        return x

# 创建模型实例
model = SimpleNet()
print(f"\n模型结构：\n{model}")

# 查看模型有多少参数
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型总参数数量：{total_params}")


print("\n" + "=" * 50)
print("第三步：定义损失函数和优化器")
print("=" * 50)

# 损失函数：用于衡量预测值和真实值的差距
criterion = nn.BCELoss()  # 二分类交叉熵损失
print("损失函数：BCELoss（二分类交叉熵）")

# 优化器：用于更新网络参数
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adam优化器，学习率0.1
print("优化器：Adam，学习率=0.1")


print("\n" + "=" * 50)
print("第四步：训练模型")
print("=" * 50)

# 记录损失值，用于后续可视化
losses = []

# 训练轮数
epochs = 1000

for epoch in range(epochs):
    # 1. 前向传播：计算预测值
    predictions = model(X)
    
    # 2. 计算损失
    loss = criterion(predictions, y)
    
    # 3. 反向传播：计算梯度
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()        # 计算新的梯度
    
    # 4. 更新参数
    optimizer.step()
    
    # 记录损失
    losses.append(loss.item())
    
    # 每100轮打印一次
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


print("\n" + "=" * 50)
print("第五步：测试模型")
print("=" * 50)

# 测试模型：让它预测所有输入
with torch.no_grad():  # 测试时不需要计算梯度
    predictions = model(X)
    print("\n输入 -> 预测值 -> 真实值")
    print("-" * 30)
    for i in range(len(X)):
        pred = predictions[i].item()
        true = y[i].item()
        result = "✓" if round(pred) == true else "✗"
        print(f"{X[i].tolist()} -> {pred:.4f} -> {int(true)} {result}")


print("\n" + "=" * 50)
print("第六步：可视化训练过程")
print("=" * 50)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.savefig('d:/pyprojects/aigc-m/training_loss.png')
print("\n✓ 损失曲线已保存到：training_loss.png")
plt.show()

print("\n" + "=" * 50)
print("训练完成！🎉")
print("=" * 50)
print("\n你刚刚完成了：")
print("1. 准备数据")
print("2. 定义神经网络")
print("3. 设置损失函数和优化器")
print("4. 训练模型（1000轮）")
print("5. 测试模型性能")
print("6. 可视化训练过程")
print("\n这就是深度学习的完整流程！")