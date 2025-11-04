# test_broadcast.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import Tensor
import numpy as np

def test_broadcast_gradient():
    print("=== 测试广播梯度 ===")
    
    # 创建需要广播的情况
    a = Tensor([[1.0, 2.0]], requires_grad=True)  # shape (1,2)
    b = Tensor([[3.0], [4.0]], requires_grad=True)  # shape (2,1)
    
    # 前向传播（会广播到 (2,2)）
    c = a + b
    print(f"a.shape: {a.shape}, b.shape: {b.shape}")
    print(f"c.shape: {c.shape}")
    
    # 反向传播
    c.backward(np.ones((2,2)))
    
    print(f"a.grad: {a.grad}")
    print(f"b.grad: {b.grad}")
    
    # 检查梯度是否正确
    # a的梯度应该沿第0轴求和：[[2.0, 2.0]]
    # b的梯度应该沿第1轴求和：[[2.0], [2.0]]
    expected_a_grad = np.array([[2.0, 2.0]])
    expected_b_grad = np.array([[2.0], [2.0]])
    
    print(f"Expected a.grad: {expected_a_grad}")
    print(f"Expected b.grad: {expected_b_grad}")
    
    if np.allclose(a.grad, expected_a_grad) and np.allclose(b.grad, expected_b_grad):
        print("✓ 广播梯度测试通过")
    else:
        print("✗ 广播梯度测试失败")

def test_matmul_gradient():
    print("\n=== 测试矩阵乘法梯度 ===")
    
    # 简单的矩阵乘法测试
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    
    c = a @ b
    print(f"a: {a.data}")
    print(f"b: {b.data}") 
    print(f"c = a @ b: {c.data}")
    
    # 反向传播
    c.backward(np.ones((2,2)))
    
    print(f"a.grad: {a.grad}")
    print(f"b.grad: {b.grad}")
    
    # 手动计算期望梯度
    # ∂L/∂A = ones(2,2) @ B^T
    expected_a_grad = np.ones((2,2)) @ b.data.T
    # ∂L/∂B = A^T @ ones(2,2)
    expected_b_grad = a.data.T @ np.ones((2,2))
    
    print(f"Expected a.grad: {expected_a_grad}")
    print(f"Expected b.grad: {expected_b_grad}")
    
    if np.allclose(a.grad, expected_a_grad) and np.allclose(b.grad, expected_b_grad):
        print("✓ 矩阵乘法梯度测试通过")
    else:
        print("✗ 矩阵乘法梯度测试失败")

if __name__ == "__main__":
    test_broadcast_gradient()
    test_matmul_gradient()