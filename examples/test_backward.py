# test_minimal.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn import Linear
from core import Tensor
import numpy as np

# 测试1: 单个线性层
print("=== 测试1: 单个线性层 ===")
linear = Linear(2, 3)
x = Tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"输入: {x.data}")
print(f"权重: {linear.weight.data}")
print(f"偏置: {linear.bias_param.data}")

y = linear(x)
print(f"输出: {y.data}")

# 测试2: 反向传播
print("\n=== 测试2: 反向传播 ===")
loss = y.sum()
print(f"Loss: {loss.data}")

loss.backward()
print(f"输入梯度: {x.grad}")
print(f"权重梯度: {linear.weight.grad}")
print(f"偏置梯度: {linear.bias_param.grad}")

# 测试3: 一步更新
print("\n=== 测试3: 参数更新 ===")
old_weight = linear.weight.data.copy()
linear.weight.data -= 0.1 * linear.weight.grad
print(f"权重更新: {old_weight} -> {linear.weight.data}")