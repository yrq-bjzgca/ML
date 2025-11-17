import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy.monitor import EnergyMonitor, SparsityMonitor, CarbonFootprintTracker, FLOPsCounter
from nn import Sequential, Linear, ReLU, Conv2d
from core.tensor import Tensor

def test_energy_monitor_basic():
    """测试基础能量监控功能"""
    print("Testing EnergyMonitor basic functionality...")
    
    # 创建简单模型
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    )
    
    # 初始化监控器
    monitor = EnergyMonitor(model)
    monitor.attach()
    
    # 模拟前向传播
    x = Tensor(np.random.randn(2, 10))
    output = model(x)
    
    # 检查监控数据
    assert len(monitor.energy_log) > 0, "Should record energy data"
    assert len(monitor.activation_stats) > 0, "Should record activation stats"
    
    # 生成报告
    report = monitor.generate_report()
    assert 'total_energy' in report, "Report should contain total energy"
    assert 'average_sparsity' in report, "Report should contain sparsity"
    assert 'total_flops' in report, "Report should contain FLOPs"
    
    # 检查具体数值
    assert report['total_energy'] >= 0, "Energy should be non-negative"
    assert 0 <= report['average_sparsity'] <= 1, "Sparsity should be between 0 and 1"
    assert report['total_flops'] >= 0, "FLOPs should be non-negative"
    
    monitor.detach()
    print("✓ EnergyMonitor basic test passed")

def test_energy_monitor_reset():
    """测试监控器重置功能"""
    print("Testing EnergyMonitor reset functionality...")
    
    model = Sequential(Linear(10, 5))
    monitor = EnergyMonitor(model)
    monitor.attach()
    
    # 第一次前向传播
    x1 = Tensor(np.random.randn(2, 10))
    output1 = model(x1)
    
    # 记录第一次的数据量
    initial_log_count = len(monitor.energy_log)
    initial_stats_count = len(monitor.activation_stats)
    
    # 重置监控器
    monitor.reset()
    
    # 检查是否重置
    assert len(monitor.energy_log) == 0, "Energy log should be empty after reset"
    assert len(monitor.activation_stats) == 0, "Activation stats should be empty after reset"
    
    # 第二次前向传播
    x2 = Tensor(np.random.randn(2, 10))
    output2 = model(x2)
    
    # 检查新数据
    assert len(monitor.energy_log) > 0, "Should record new energy data"
    assert len(monitor.activation_stats) > 0, "Should record new activation stats"
    
    monitor.detach()
    print("✓ EnergyMonitor reset test passed")

def test_sparsity_monitor():
    """测试稀疏度监控器"""
    print("Testing SparsityMonitor...")
    
    sparsity_monitor = SparsityMonitor()
    
    # 测试不同稀疏度的激活
    zero_activation = Tensor(np.zeros((5, 5)))
    ones_activation = Tensor(np.ones((5, 5)))
    mixed_activation = Tensor(np.array([[0, 1, 0], [1, 0, 1]]))
    
    # 记录激活
    sparsity_monitor.record_activation("layer1", zero_activation)
    sparsity_monitor.record_activation("layer2", ones_activation)
    sparsity_monitor.record_activation("layer3", mixed_activation)
    
    # 检查稀疏度计算
    assert sparsity_monitor.get_layer_sparsity("layer1") == 1.0, "Zero activation should have sparsity 1.0"
    assert sparsity_monitor.get_layer_sparsity("layer2") == 0.0, "Ones activation should have sparsity 0.0"
    assert abs(sparsity_monitor.get_layer_sparsity("layer3") - 0.5) < 1e-6, "Mixed activation should have sparsity 0.5"
    
    # 检查整体稀疏度
    overall_sparsity = sparsity_monitor.get_overall_sparsity()
    assert 0 <= overall_sparsity <= 1, "Overall sparsity should be between 0 and 1"
    
    # 检查最稀疏的层
    most_sparse = sparsity_monitor.get_most_sparse_layers()
    assert len(most_sparse) > 0, "Should return most sparse layers"
    
    # 生成报告
    report = sparsity_monitor.generate_report()
    assert 'overall_sparsity' in report, "Report should contain overall sparsity"
    
    print("✓ SparsityMonitor test passed")

def test_carbon_footprint_tracker():
    """测试碳足迹追踪器"""
    print("Testing CarbonFootprintTracker...")
    
    tracker = CarbonFootprintTracker(carbon_intensity=0.5, power_draw_watts=200.0)
    
    # 测试训练能耗
    tracker.start_training()
    time.sleep(0.1)  # 模拟训练时间
    tracker.end_training()
    
    # 测试推理能耗
    tracker.record_inference(flops=1e9, duration=1.0)  # 1GFLOPs, 1秒
    
    # 检查碳足迹计算
    training_carbon = tracker.get_training_carbon()
    inference_carbon = tracker.get_inference_carbon()
    total_carbon = tracker.get_total_carbon_footprint()
    
    assert training_carbon >= 0, "Training carbon should be non-negative"
    assert inference_carbon >= 0, "Inference carbon should be non-negative"
    assert total_carbon >= 0, "Total carbon should be non-negative"
    assert abs(total_carbon - (training_carbon + inference_carbon)) < 1e-6, "Total carbon should equal sum of training and inference"
    
    # 生成报告
    report = tracker.generate_report()
    assert 'total_carbon_kg' in report, "Report should contain total carbon"
    assert 'training_carbon_kg' in report, "Report should contain training carbon"
    assert 'inference_carbon_kg' in report, "Report should contain inference carbon"
    
    print("✓ CarbonFootprintTracker test passed")

def test_flops_counter():
    """测试FLOPs计数器"""
    print("Testing FLOPsCounter...")
    
    counter = FLOPsCounter()
    
    # 测试Linear层FLOPs
    class MockLinear:
        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features
    
    linear_layer = MockLinear(100, 50)
    input_shape = (32, 100)  # batch_size=32
    output_shape = (32, 50)
    
    linear_flops = counter.estimate_layer_flops(linear_layer, input_shape, output_shape)
    expected_linear_flops = 2 * 100 * 50 * 32  # 2 * in * out * batch_size
    assert linear_flops == expected_linear_flops, f"Linear FLOPs incorrect: {linear_flops} vs {expected_linear_flops}"
    
    # 测试Conv2d层FLOPs
    class MockConv2d:
        def __init__(self, in_channels, out_channels, kernel_size):
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
    
    conv_layer = MockConv2d(3, 16, (3, 3))
    input_shape = (8, 3, 32, 32)  # batch, channels, height, width
    output_shape = (8, 16, 30, 30)  # 32-3+1=30
    
    conv_flops = counter.estimate_layer_flops(conv_layer, input_shape, output_shape)
    expected_conv_flops = 2 * 3 * 3 * 3 * 16 * 30 * 30 * 8  # 2 * in_c * k_h * k_w * out_c * out_h * out_w * batch
    assert conv_flops == expected_conv_flops, f"Conv2d FLOPs incorrect: {conv_flops} vs {expected_conv_flops}"
    
    print("✓ FLOPsCounter test passed")

def test_energy_monitor_integration():
    """测试能量监控器集成功能"""
    print("Testing EnergyMonitor integration...")
    
    # 创建更复杂的模型
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10)
    )
    
    # 初始化所有监控器
    energy_monitor = EnergyMonitor(model)
    sparsity_monitor = SparsityMonitor()
    carbon_tracker = CarbonFootprintTracker()
    
    energy_monitor.attach()
    carbon_tracker.start_training()
    
    # 模拟训练过程
    for i in range(3):
        x = Tensor(np.random.randn(32, 784))
        output = model(x)
        
        # 记录稀疏度
        for layer_name, stats in energy_monitor.activation_stats.items():
            # 这里简化处理，实际应该使用真实的激活值
            mock_activation = Tensor(np.random.randn(32, 128) * 0.1)
            sparsity_monitor.record_activation(layer_name, mock_activation)
    
    carbon_tracker.end_training()
    energy_monitor.detach()
    
    # 检查所有监控器都正常工作
    energy_report = energy_monitor.generate_report()
    sparsity_report = sparsity_monitor.generate_report()
    carbon_report = carbon_tracker.generate_report()
    
    assert energy_report['total_energy'] > 0, "Should have positive energy consumption"
    assert sparsity_report['overall_sparsity'] >= 0, "Should have valid sparsity"
    assert carbon_report['total_carbon_kg'] >= 0, "Should have valid carbon footprint"
    
    print("✓ EnergyMonitor integration test passed")

def run_all_monitor_tests():
    """运行所有monitor测试"""
    print("Running Energy Monitor tests...")
    print("=" * 50)
    
    test_energy_monitor_basic()
    test_energy_monitor_reset()
    test_sparsity_monitor()
    test_carbon_footprint_tracker()
    test_flops_counter()
    test_energy_monitor_integration()
    
    print("=" * 50)
    print("All Energy Monitor tests passed! ✅")

if __name__ == "__main__":
    run_all_monitor_tests()