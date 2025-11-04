# test_training_flow.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn import Sequential, Linear, ReLU
from core import Tensor, cross_entropy
from core.optim import SGD
import numpy as np

def test_training_flow():
    print("=== 测试训练流程 ===")
    
    # 创建简单模型
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 2)
    )
    
    # 小批量数据
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]], requires_grad=False)
    y = Tensor([0, 1, 0])  # 类别标签
    
    print("初始参数:")
    for name, param in model.named_parameters():
        print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    
    opt = SGD(model.parameters(), lr=0.01)
    
    # 训练几步并监控
    for step in range(100):
        print(f"\n--- 第 {step+1} 步 ---")
        
        # 前向传播
        logits = model(x)
        print(f"Logits范围: [{logits.data.min():.4f}, {logits.data.max():.4f}]")
        
        loss = cross_entropy(logits, y)
        print(f"Loss: {loss.data:.4f}")
        
        # 反向传播前检查梯度
        print("反向传播前梯度:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad_norm={np.linalg.norm(param.grad):.4f}")
            else:
                print(f"{name}: 梯度为None")
        
        # 反向传播
        opt.zero_grad()
        loss.backward()
        
        # 反向传播后检查梯度
        print("反向传播后梯度:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = np.linalg.norm(param.grad.reshape(-1))
                print(f"{name}: grad_norm={grad_norm:.4f}")
            else:
                print(f"{name}: 梯度为None")
        
        # 参数更新
        opt.step()
        
        print("更新后参数:")
        for name, param in model.named_parameters():
            print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")

def test_parameter_growth():
    print("\n=== 测试参数增长 ===")
    
    # 创建一个简单的线性层并多次更新
    linear = Linear(10, 5)
    x = Tensor(np.random.randn(32, 10) * 0.1)  # 小输入
    
    print("初始权重统计:")
    print(f"权重范围: [{linear.weight.data.min():.4f}, {linear.weight.data.max():.4f}]")
    print(f"权重均值: {linear.weight.data.mean():.4f}")
    print(f"权重标准差: {linear.weight.data.std():.4f}")
    
    opt = SGD([linear.weight, linear.bias_param], lr=0.01)
    
    for i in range(100):
        # 模拟一个总是为正的梯度（这会导致参数持续增长）
        linear.weight.grad = np.ones_like(linear.weight.data) * 0.1
        linear.bias_param.grad = np.ones_like(linear.bias_param.data) * 0.1
        
        opt.step()
        opt.zero_grad()
        
        if i % 2 == 0:
            print(f"第{i}步后权重范围: [{linear.weight.data.min():.4f}, {linear.weight.data.max():.4f}]")

if __name__ == "__main__":
    test_training_flow()
    test_parameter_growth()