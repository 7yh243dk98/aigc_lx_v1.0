import torch

def f(a):
    """一个包含循环和分支的复杂函数"""
    b = a * 2
    # while 循环：循环次数取决于 a 的值
    while b.norm() < 1000:
        b = b * 2
    
    # if 分支：走哪条路取决于 b 的值
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# 1. 创建一个随机的标量输入 a
# size=() 表示是一个 0 维张量（即一个孤立的数字）
a = torch.randn(size=(), requires_grad=True)

# 2. 运行函数并计算结果
d = f(a)

# 3. 反向传播计算梯度
d.backward()

# 4. 打印结果进行验证
print(f"输入 a: {a.item():.4f}")
print(f"输出 d: {d.item():.4f}")
print(f"计算出的梯度 a.grad: {a.grad.item():.4f}")
print(f"理论上的斜率 d / a: {(d / a).item():.4f}")

# 5. 验证两者是否相等
is_correct = torch.allclose(a.grad, d / a)
print(f"\n验证结果 (a.grad == d / a): {is_correct}")

if is_correct:
    print("\n✅ 验证成功！说明 PyTorch 完美追踪了循环和分支中的所有倍数关系。")
else:
    print("\n❌ 验证失败，请检查逻辑。")
