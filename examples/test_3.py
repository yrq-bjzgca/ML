# test_numerical_stability.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import Tensor
from core.functional import log_softmax, cross_entropy
import numpy as np

def test_large_logits():
    print("=== 测试大logits的数值稳定性 ===")
    
    # 测试极端大的logits
    large_logits = Tensor([[100.0, 200.0, 300.0]], requires_grad=True)
    print(f"输入logits: {large_logits.data}")
    
    log_p = log_softmax(large_logits, axis=-1)
    print(f"log_softmax输出: {log_p.data}")
    
    # 检查是否合理
    exp_log_p = np.exp(log_p.data)
    print(f"exp(log_softmax): {exp_log_p}")
    print(f"和: {np.sum(exp_log_p)}")  # 应该接近1.0
    
    # 测试梯度
    targets = Tensor([2])
    loss = cross_entropy(large_logits, targets)
    print(f"Loss: {loss.data}")
    
    loss.backward()
    print(f"梯度范围: [{large_logits.grad.min():.6f}, {large_logits.grad.max():.6f}]")

def test_log_softmax_gradient():
    print("\n=== 测试log_softmax梯度 ===")
    
    # 创建简单的logits
    logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    print(f"输入logits: {logits.data}")
    
    # 前向传播
    log_p = log_softmax(logits, axis=-1)
    print(f"log_softmax输出: {log_p.data}")
    
    # 手动设置上游梯度
    log_p.grad = np.ones_like(log_p.data)
    
    # 触发反向传播
    log_p._backward()
    
    print(f"logits梯度: {logits.grad}")
    
    # 手动验证梯度
    # log_softmax(x_i) = x_i - log(sum(exp(x)))
    # 梯度公式: ∂L/∂x_i = ∂L/∂log_p_i - exp(log_p_i) * sum(∂L/∂log_p)
    manual_grad = np.ones_like(log_p.data) - np.exp(log_p.data) * np.sum(np.ones_like(log_p.data))
    print(f"手动计算梯度: {manual_grad}")
    
    if np.allclose(logits.grad, manual_grad):
        print("✓ log_softmax梯度正确")
    else:
        print("✗ log_softmax梯度错误")

if __name__ == "__main__":
    test_large_logits()
    test_log_softmax_gradient()