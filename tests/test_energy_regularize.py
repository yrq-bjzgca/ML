import numpy as np
import sys
import os

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

def get_model_layers(model):
    """安全地获取模型层列表，处理不同的模型结构"""
    if hasattr(model, 'layers'):
        return model.layers
    elif hasattr(model, '_layers'):
        return model._layers
    elif hasattr(model, '_modules'):
        # 处理使用_modules的模型结构
        return list(model._modules.values())
    else:
        # 如果都没有，返回只包含模型本身的列表
        return [model]

class TestEnergyAwareRegularizer:
    """测试能量感知正则化器"""
    
    def test_energy_aware_regularizer_basic(self):
        """测试基础能量感知正则化"""
        regularizer = EnergyAwareRegularizer(
            energy_coeff=1e-8, 
            sparsity_coeff=1e-3
        )
        
        # 创建简单模型
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        
        # 安全地获取层并设置参数
        layers = get_model_layers(model)
        
        # 设置权重和偏置
        layers['0'].weight.data = np.ones_like(layers['0'].weight.data)
        layers['0'].bias.data = np.ones_like(layers['0'].bias.data)
        layers['2'].weight.data = np.ones_like(layers['2'].weight.data)
        layers['2'].bias.data = np.ones_like(layers['2'].bias.data)
        
        # 创建模拟激活字典
        activations_dict = {
            'Linear1': Tensor(np.random.randn(32, 20) * 0.1),
            'ReLU1': Tensor(np.random.randn(32, 20).clip(0, 1)),
            'Linear2': Tensor(np.random.randn(32, 5) * 0.1)
        }
        
        # 计算正则化损失
        loss = regularizer(model, activations_dict)
        
        # 验证返回值为浮点数且非负
        assert isinstance(loss, float), "EnergyAwareRegularizer should return a float"
        assert loss >= 0, "Regularization loss should be non-negative"
        
        print("✓ EnergyAwareRegularizer basic test passed")
    
    def test_get_energy_factor(self):
        """测试能量系数计算"""
        regularizer = EnergyAwareRegularizer()
        
        # 测试不同形状的参数
        test_cases = [
            # (参数形状, 预期能量系数)
            ((3, 3, 3, 3), 2.0),  # 卷积核
            ((100, 50), 1.0),     # 全连接权重
            ((10,), 0.1),         # 偏置
            ((5, 5, 5, 5, 5), 0.1)  # 未知形状，默认
        ]
        
        for shape, expected_factor in test_cases:
            # 创建模拟参数张量
            param = Tensor(np.ones(shape))
            factor = regularizer._get_energy_factor(param)
            
            assert factor == expected_factor, \
                f"Energy factor incorrect for shape {shape}: {factor} vs {expected_factor}"
        
        print("✓ Energy factor calculation test passed")
    
    def test_sparsity_calculation(self):
        """测试稀疏度计算"""
        regularizer = EnergyAwareRegularizer()
        
        test_cases = [
            # (activation_data, expected_sparsity)
            (np.zeros((10, 10)), 1.0),      # 全零，稀疏度=1
            (np.ones((10, 10)), 0.0),       # 全非零，稀疏度=0
            (np.array([[0, 1, 0], [1, 0, 1]]), 0.5),  # 一半为零
            # (np.random.randn(5, 5) * 1e-7, 1.0),  # 接近零的值
        ]
        
        for activation_data, expected_sparsity in test_cases:
            sparsity = regularizer._calculate_sparsity(activation_data)
            
            # 允许小的浮点误差
            assert abs(sparsity - expected_sparsity) < 1e-6, \
                f"Sparsity calculation incorrect: {sparsity} vs {expected_sparsity}"
        
        print("✓ Sparsity calculation test passed")
    
    def test_energy_aware_with_tensor_input(self):
        """测试使用Tensor输入的稀疏度计算"""
        regularizer = EnergyAwareRegularizer()
        
        # 使用Tensor作为输入
        zero_tensor = Tensor(np.zeros((5, 5)))
        ones_tensor = Tensor(np.ones((5, 5)))
        mixed_tensor = Tensor(np.array([[0, 1], [1, 0]]))
        
        zero_sparsity = regularizer._calculate_sparsity(zero_tensor)
        ones_sparsity = regularizer._calculate_sparsity(ones_tensor)
        mixed_sparsity = regularizer._calculate_sparsity(mixed_tensor)
        
        assert zero_sparsity == 1.0, f"Zero tensor sparsity should be 1.0, got {zero_sparsity}"
        assert ones_sparsity == 0.0, f"Ones tensor sparsity should be 0.0, got {ones_sparsity}"
        assert mixed_sparsity == 0.5, f"Mixed tensor sparsity should be 0.5, got {mixed_sparsity}"
        
        print("✓ Sparsity calculation with Tensor input test passed")
    
    def test_energy_aware_coefficients(self):
        """测试不同系数的能量感知正则化"""
        model = Sequential(Linear(10, 5))
        
        # 安全地获取层并设置参数
        layers = get_model_layers(model)
        layers['0'].weight.data = np.ones_like(layers['0'].weight.data)
        layers['0'].bias.data = np.ones_like(layers['0'].bias.data)
        
        # 测试不同系数组合
        test_cases = [
            (1e-8, 1e-3),  # 基础系数
            (1e-7, 1e-4),  # 高能量系数，低稀疏系数
            (1e-9, 1e-2),  # 低能量系数，高稀疏系数
        ]
        
        for energy_coeff, sparsity_coeff in test_cases:
            regularizer = EnergyAwareRegularizer(
                energy_coeff=energy_coeff,
                sparsity_coeff=sparsity_coeff
            )
            
            loss = regularizer(model, {})
            
            # 验证返回值为浮点数
            assert isinstance(loss, float), "Should return float for all coefficient combinations"
        
        print("✓ EnergyAwareRegularizer coefficient test passed")
    
    def test_energy_estimation_with_conv_layers(self):
        """测试包含卷积层的能量估计"""
        # 创建包含卷积层的模型
        model = Sequential(
            Conv2d(3, 16, kernel_size=3, padding=1),
            ReLU(),
            Linear(16 * 32 * 32, 10)  # 假设输入是32x32
        )
        
        # 安全地获取层并设置参数
        layers = get_model_layers(model)
        
        layers['0'].weight.data = np.ones_like(layers['0'].weight.data)
        if hasattr(layers['0'], 'bias') and layers['0'].bias is not None:
            layers['0'].bias.data = np.ones_like(layers['0'].bias.data)
        
        layers['2'].weight.data = np.ones_like(layers['2'].weight.data)
        layers['2'].bias.data = np.ones_like(layers['2'].bias.data)
        
        regularizer = EnergyAwareRegularizer()
        loss = regularizer(model, {})
        
        assert isinstance(loss, float) and loss >= 0, \
            "Energy estimation should work with conv layers"
        
        print("✓ Energy estimation with conv layers test passed")
    
    def test_energy_aware_without_activations(self):
        """测试没有激活字典的情况"""
        model = Sequential(Linear(10, 5))
        
        # 安全地获取层并设置参数
        layers = get_model_layers(model)
        layers['0'].weight.data = np.ones_like(layers['0'].weight.data)
        layers['0'].bias.data = np.ones_like(layers['0'].bias.data)
        
        regularizer = EnergyAwareRegularizer()
        
        # 不传递activations_dict
        loss_without_activations = regularizer(model)
        
        # 传递空的activations_dict
        loss_with_empty_activations = regularizer(model, {})
        
        # 两种情况应该产生相同的结果
        assert abs(loss_without_activations - loss_with_empty_activations) < 1e-6, \
            "Should handle None and empty activations_dict the same way"
        
        print("✓ EnergyAwareRegularizer without activations test passed")

def test_energy_aware_regularizer_complete():
    """完整的能量感知正则化器测试"""
    tester = TestEnergyAwareRegularizer()
    
    tester.test_energy_aware_regularizer_basic()
    tester.test_get_energy_factor()
    tester.test_sparsity_calculation()
    tester.test_energy_aware_with_tensor_input()
    tester.test_energy_aware_coefficients()
    tester.test_energy_estimation_with_conv_layers()
    tester.test_energy_aware_without_activations()
    
    print("All EnergyAwareRegularizer tests passed! ✅")

if __name__ == "__main__":
    test_energy_aware_regularizer_complete()