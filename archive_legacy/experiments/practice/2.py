import torch

# ... 前面的数据准备保持不变 ...
X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = torch.tensor([0.0, 1.0, 1.0, 1.0])
m = len(X)
# 2. 初始化参数
w = torch.tensor([0.0, 0.0])
b = 0.0
learning_rate = 0.5  # 稍微加大一点学习率
epochs = 1000        # 重点：增加迭代次数

# 3. 训练逻辑 - 增加外层循环
for epoch in range(epochs):
    # 每一轮开始前，梯度要清零
    J = 0.0
    dw1, dw2, db = 0.0, 0.0, 0.0
    
    for i in range(m):
        # 前向传播
        z = w @ X[i] + b
        a_val = 1 / (1 + torch.exp(-z))
        
        # 损失计算
        J += -(y[i] * torch.log(a_val + 1e-8) + (1 - y[i]) * torch.log(1 - a_val + 1e-8))
        
        # 梯度累加
        dz = a_val - y[i]
        dw1 += X[i, 0] * dz
        dw2 += X[i, 1] * dz
        db += dz

    # 计算平均梯度并更新
    dw1 /= m
    dw2 /= m
    db /= m
    w[0] -= learning_rate * dw1
    w[1] -= learning_rate * dw2
    b -= learning_rate * db
    
    # 每 100 轮打印一次进度
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {J/m:.4f}")

# 4. 预测
test_input = torch.tensor([1.0, 0.0])
z_test = w @ test_input + b
prediction = 1 / (1 + torch.exp(-z_test))
print(f"\n训练 {epochs} 轮后的 [1,0] 预测概率: {prediction.item():.4f}")