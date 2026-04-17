from re import M
import torch
import numpy as np

# n1 = np.array([1,2,3])
# X = torch.from_numpy(n1)
# print(X)

# # 创建一个需要梯度的张量
# tensor_requires_grad = torch.tensor([2.0], requires_grad=True)

# # 进行一些操作
# tensor_result = tensor_requires_grad * 2
# tensor_result2 = tensor_result * 3
# # 计算梯度
# tensor_result.backward(retain_graph=True)
# tensor_result2.backward()
# print(tensor_requires_grad.grad,tensor_result2.grad)  # 输出梯度





### 🛠️ 修正后的代码（你可以直接运行对比）：


# import torch

# # 创建叶子节点
# x = torch.tensor([2.0], requires_grad=True)

# # 操作
# y = x * 2
# z = y * 3

# # 让中间节点也保留梯度
# y.retain_grad()
# z.retain_grad()

# # 第一次计算：y 对 x 的梯度
# y.backward(retain_graph=True)
# print(f"第一次后 x 的梯度 (dy/dx): {x.grad.item()}") # 应该是 2.0

# # 清零梯度，否则第二次会累加
# x.grad.zero_() 

# # 第二次计算：z 对 x 的梯度
# z.backward()
# print(f"第二次后 x 的梯度 (dz/dx): {x.grad.item()}") # 应该是 6.0

# # 查看中间节点 z 自己的梯度 (dz/dz)
# print(f"z 自己的梯度: {z.grad.item()}") # 应该是 1.0
# #-------------------------------------------------------------
# # 29行：定义一维张量
# x = torch.tensor([2.0, 3.0], requires_grad=True)

# y = x * 2
# z = y * 3

# y.retain_grad()
# z.retain_grad()

# # 40行：求和后再 backward，这样就变成了标量
# y.sum().backward(retain_graph=True)

# # 41行：去掉 .item()，因为 x.grad 现在有两个值
# print(f"第一次后 x 的梯度 (dy/dx): {x.grad}") 

# x.grad.zero_() 

# # 47行：同样求和后再 backward
# z.sum().backward()
# print(f"第二次后 x 的梯度 (dz/dx): {x.grad}")

import torch
import numpy as np

# 1. 准备数据 (使用浮点数避免计算错误)
X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = torch.tensor([0.0, 1.0, 1.0, 1.0])
m = len(X)

# 2. 初始化参数 (使用浮点数)
w = torch.tensor([0.0, 0.0], requires_grad=False)
b = 0.0
learning_rate = 0.1

# 3. 训练逻辑 (这里只跑一个 epoch 作为演示)
J = 0.0
dw1, dw2, db = 0.0, 0.0, 0.0

for i in range(m):
    # 前向传播
    z = w @ X[i] + b # 使用点积
    a_val = 1 / (1 + torch.exp(-z)) # 正确的 Sigmoid
    
    # 计算损失 (添加微小值防止 log(0))
    J += -(y[i] * torch.log(a_val + 1e-8) + (1 - y[i]) * torch.log(1 - a_val + 1e-8))
    
    # 计算梯度
    dz = a_val - y[i]
    dw1 += X[i, 0] * dz
    dw2 += X[i, 1] * dz
    db += dz

# --- 循环结束后再计算平均值和更新 ---
J /= m
dw1 /= m
dw2 /= m
db /= m

# 更新参数
w[0] = w[0] - learning_rate * dw1
w[1] = w[1] - learning_rate * dw2
b = b - learning_rate * db

print(f"损失值 J: {J:.4f}")
print(f"更新后的 w: {w}, b: {b:.4f}")

# 4. 预测
test_input = torch.tensor([1.0, 0.0])
z_test = torch.dot(w, test_input) + b
prediction = 1 / (1 + torch.exp(-z_test))
print(f"\n[1,0] 的预测概率: {prediction.item():.4f},{prediction}")