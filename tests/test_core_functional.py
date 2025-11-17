
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy.regularize import (
    L2Regularizer, 
    FLOPsCalculator,
    FLOPsRegularizer,
    EnergyAwareRegularizer,
    CombinedRegularizer
)
from nn import Sequential, Linear, ReLU, Conv2d
from core.tensor import Tensor
from core.optim import SGD, AdaGrad,Momentum,Adam

from core.functional import relu,sigmoid,tanh,softmax,log_softmax,cross_entropy,conv2d,nll_loss,mse_loss

# 假设你的Tensor类已经定义在上面
# 这里只展示测试代码

def test_activation_functions():
    print("===== 激活函数测试 =====")
    
    # 测试ReLU
    print("1. ReLU测试")
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = relu(x)
    y.backward(np.ones_like(y.data))
    
    print(f"输入: {x.data}")
    print(f"ReLU输出: {y.data}")
    print(f"梯度: {x.grad}")
    expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    assert np.allclose(x.grad, expected_grad), f"ReLU梯度错误: {x.grad} != {expected_grad}"
    print("ReLU测试通过 ✓")
    
    # 测试Sigmoid
    print("\n2. Sigmoid测试")
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = sigmoid(x)
    y.backward(np.ones_like(y.data))
    
    print(f"输入: {x.data}")
    print(f"Sigmoid输出: {y.data}")
    print(f"梯度: {x.grad}")
    
    # 手动计算期望梯度
    sigmoid_output = 1 / (1 + np.exp(-x.data))
    expected_grad = sigmoid_output * (1 - sigmoid_output)
    assert np.allclose(x.grad, expected_grad, atol=1e-6), f"Sigmoid梯度错误: {x.grad} != {expected_grad}"
    print("Sigmoid测试通过 ✓")
    
    # 测试Tanh
    print("\n3. Tanh测试")
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = tanh(x)
    y.backward(np.ones_like(y.data))
    
    print(f"输入: {x.data}")
    print(f"Tanh输出: {y.data}")
    print(f"梯度: {x.grad}")
    
    # 手动计算期望梯度
    tanh_output = np.tanh(x.data)
    expected_grad = 1 - tanh_output ** 2
    assert np.allclose(x.grad, expected_grad, atol=1e-6), f"Tanh梯度错误: {x.grad} != {expected_grad}"
    print("Tanh测试通过 ✓")

def test_softmax():
    print("\n===== Softmax测试 =====")
    
    # 测试1: 基础softmax
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = softmax(x, axis=-1)
    y.backward(np.ones_like(y.data))
    
    print(f"输入: {x.data}")
    print(f"Softmax输出: {y.data}")
    print(f"输出和: {y.data.sum()}")
    print(f"梯度: {x.grad}")
    
    # 检查输出和为1
    assert abs(y.data.sum() - 1.0) < 1e-6, "Softmax输出和不为1"
    print("Softmax基础测试通过 ✓")
    
    # 测试2: 数值稳定性
    x = Tensor([[1000.0, 1000.0, 1000.0]], requires_grad=True)
    y = softmax(x, axis=-1)
    
    print(f"大数值输入: {x.data}")
    print(f"Softmax输出: {y.data}")
    assert not np.any(np.isnan(y.data)), "Softmax数值不稳定"
    print("Softmax数值稳定性测试通过 ✓")

def test_log_softmax():
    print("\n===== Log Softmax测试 =====")
    
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = log_softmax(x, axis=-1)
    y.backward(np.ones_like(y.data))
    
    print(f"输入: {x.data}")
    print(f"Log Softmax输出: {y.data}")
    print(f"梯度: {x.grad}")
    
    # 验证 log_softmax(x) = log(softmax(x))
    softmax_out = softmax(x, axis=-1)
    expected = np.log(softmax_out.data + 1e-12)
    assert np.allclose(y.data, expected, atol=1e-6), "Log Softmax输出错误"
    print("Log Softmax测试通过 ✓")

def test_loss_functions():
    print("\n===== 损失函数测试 =====")
    
    # 测试NLL Loss
    print("1. NLL Loss测试")
    log_probs = Tensor([
        [-1.0, -2.0, -3.0],  # 真实类在位置0
        [-3.0, -1.0, -2.0]   # 真实类在位置1
    ], requires_grad=True)
    targets = Tensor([0, 1])  # 类别索引
    
    loss = nll_loss(log_probs, targets)
    loss.backward()
    
    print(f"Log概率: {log_probs.data}")
    print(f"目标: {targets.data}")
    print(f"NLL Loss: {loss.data}")
    print(f"Log概率梯度: {log_probs.grad}")
    
    # 手动验证
    selected = np.array([log_probs.data[0, 0], log_probs.data[1, 1]])
    expected_loss = -selected.mean()
    assert abs(loss.data - expected_loss) < 1e-6, "NLL Loss计算错误"
    print("NLL Loss测试通过 ✓")
    
    # 测试Cross Entropy
    print("\n2. Cross Entropy测试")
    logits = Tensor([
        [2.0, 1.0, 0.1],  # 真实类在位置0
        [0.1, 2.0, 0.1]   # 真实类在位置1
    ], requires_grad=True)
    targets = Tensor([0, 1])
    
    loss = cross_entropy(logits, targets)
    loss.backward()
    
    print(f"Logits: {logits.data}")
    print(f"目标: {targets.data}")
    print(f"Cross Entropy Loss: {loss.data}")
    print(f"Logits梯度: {logits.grad}")
    
    # 验证梯度形状
    assert logits.grad.shape == logits.data.shape, "Cross Entropy梯度形状错误"
    print("Cross Entropy测试通过 ✓")
    
    # 测试MSE Loss
    print("\n3. MSE Loss测试")
    pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 1.8, 2.9])
    
    loss = mse_loss(pred, target)
    loss.backward()
    
    print(f"预测: {pred.data}")
    print(f"目标: {target.data}")
    print(f"MSE Loss: {loss.data}")
    print(f"预测梯度: {pred.grad}")
    
    # 手动验证
    expected_loss = ((pred.data - target.data) ** 2).mean()
    assert abs(loss.data - expected_loss) < 1e-6, "MSE Loss计算错误"
    print("MSE Loss测试通过 ✓")

def test_complex_chain():
    print("\n===== 复杂链式测试 =====")
    
    # 构建一个简单的神经网络前向传播
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    w = Tensor([[0.5, -0.5], [0.1, 0.9]], requires_grad=True)
    b = Tensor([0.1, 0.2], requires_grad=True)
    
    # 前向传播
    linear = x @ w + b  # 矩阵乘法 + 偏置
    activated = relu(linear)  # ReLU激活
    normalized = softmax(activated, axis=-1)  # Softmax归一化
    
    # 计算损失
    targets = Tensor([0, 1])  # 两个样本的真实类别
    loss = cross_entropy(normalized, targets)
    
    print(f"输入形状: {x.shape}")
    print(f"线性输出形状: {linear.shape}")
    print(f"激活输出形状: {activated.shape}")
    print(f"归一化输出形状: {normalized.shape}")
    print(f"损失: {loss.data}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度是否存在
    assert x.grad is not None, "输入梯度未计算"
    assert w.grad is not None, "权重梯度未计算"
    assert b.grad is not None, "偏置梯度未计算"
    
    print(f"输入梯度形状: {x.grad.shape}")
    print(f"权重梯度形状: {w.grad.shape}")
    print(f"偏置梯度形状: {b.grad.shape}")
    
    print("复杂链式测试通过 ✓")


# 运行所有测试
test_activation_functions()
test_softmax()
test_log_softmax()
test_loss_functions()
test_complex_chain()

print("\n🎉 所有测试通过！")