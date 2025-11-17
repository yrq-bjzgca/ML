import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tensor import Tensor
from nn import Linear, Sequential
from energy.monitor import EnergyMonitor

def test_tensor_name_basic():
    """测试基础Tensor名称功能"""
    print("Testing basic tensor name functionality...")
    
    # 测试自动命名
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    assert t1.name.startswith("tensor_"), f"Expected auto name, got {t1.name}"
    assert t2.name.startswith("tensor_"), f"Expected auto name, got {t2.name}"
    assert t1.name != t2.name, "Auto names should be unique"
    
    # 测试自定义命名
    t3 = Tensor([7, 8, 9], name="custom_tensor")
    assert t3.name == "custom_tensor", f"Expected custom name, got {t3.name}"
    
    print("✓ Basic tensor name functionality works")

def test_tensor_name_in_operations():
    """测试运算中的Tensor名称"""
    print("Testing tensor names in operations...")
    
    a = Tensor([1, 2], name="A")
    b = Tensor([3, 4], name="B")
    
    # 测试加法
    c = a + b
    assert "add" in c.name and "A" in c.name and "B" in c.name, f"Unexpected add name: {c.name}"
    
    # 测试乘法
    d = a * b
    assert "mul" in d.name and "A" in d.name and "B" in d.name, f"Unexpected mul name: {d.name}"
    

    print("✓ Tensor names in operations work correctly")

def test_layer_parameter_names():
    """测试网络层参数名称"""
    print("Testing layer parameter names...")
    
    linear = Linear(10, 5)
    
    # 检查权重名称
    assert "Linear_weight" in linear.weight.name, f"Unexpected weight name: {linear.weight.name}"
    assert "10x5" in linear.weight.name, f"Weight name should contain dimensions: {linear.weight.name}"
    
    # 检查偏置名称
    assert "Linear_bias" in linear.bias.name, f"Unexpected bias name: {linear.bias.name}"
    assert "5" in linear.bias.name, f"Bias name should contain output features: {linear.bias.name}"
    
    print("✓ Layer parameter names work correctly")

def test_energy_monitor_with_names():
    """测试能量监控器与名称功能的集成"""
    print("Testing energy monitor with tensor names...")
    
    model = Sequential(
        Linear(10, 20),
        Linear(20, 5)
    )
    
    monitor = EnergyMonitor(model)
    monitor.attach()
    
    # 前向传播一次以收集数据
    x = Tensor(np.random.randn(2, 10), name="input_data")
    output = model(x)
    
    # 检查是否记录了张量名称
    assert len(monitor.tensor_names) > 0, "Should record tensor names"
    
    # 检查是否能找到特定模式的张量
    linear_tensors = monitor.find_tensors_by_pattern("linear")
    assert len(linear_tensors) > 0, "Should find tensors with 'linear' in name"
    
    # 生成详细报告
    report = monitor.generate_detailed_report()
    assert 'tensor_statistics' in report, "Detailed report should include tensor statistics"
    
    monitor.detach()
    print("✓ Energy monitor with tensor names works correctly")

def test_tensor_name_in_backward():
    """测试反向传播中的Tensor名称"""
    print("Testing tensor names in backward pass...")
    
    a = Tensor([1.0, 2.0], requires_grad=True, name="param_A")
    b = Tensor([3.0, 4.0], requires_grad=True, name="param_B")
    
    # 前向传播
    c = a * b
    d = c.sum()
    
    # 反向传播
    d.backward()
    
    # 检查梯度张量的名称（虽然梯度通常不命名，但原始参数名称应该保留）
    assert a.name == "param_A", "Parameter name should persist through backward"
    assert b.name == "param_B", "Parameter name should persist through backward"
    
    print("✓ Tensor names persist through backward pass")

def test_static_methods_with_names():
    """测试静态方法中的名称功能"""
    print("Testing static methods with names...")
    
    # 测试zeros
    zeros_tensor = Tensor.zeros((3, 3), name="zeros_matrix")
    assert zeros_tensor.name == "zeros_matrix", f"Unexpected zeros name: {zeros_tensor.name}"
    
    # 测试ones
    ones_tensor = Tensor.ones((2, 2), name="ones_matrix")
    assert ones_tensor.name == "ones_matrix", f"Unexpected ones name: {ones_tensor.name}"
    
    # 测试randn
    randn_tensor = Tensor.randn((4, 4), name="random_matrix")
    assert randn_tensor.name == "random_matrix", f"Unexpected randn name: {randn_tensor.name}"
    
    print("✓ Static methods with names work correctly")

if __name__ == "__main__":
    test_tensor_name_basic()
    test_tensor_name_in_operations()
    test_layer_parameter_names()
    test_energy_monitor_with_names()
    test_tensor_name_in_backward()
    test_static_methods_with_names()
    print("All tensor name tests passed! ✅")