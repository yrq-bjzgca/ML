import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy.pruning import (
    EnergyAwarePruner, 
    MagnitudeEnergyPruner,
    GradientEnergyPruner,
    StructuredEnergyPruner,
    ProgressiveEnergyPruner
)
from nn import Sequential, Linear, ReLU, Conv2d
from core.tensor import Tensor

class MockEnergyMonitor:
    """模拟能量监控器用于测试"""
    def __init__(self):
        self.activation_stats = {
            'layer_0_Linear': {'energy': 1000000},
            'layer_2_Linear': {'energy': 2000000},
            'conv_layer': {'energy': 5000000}
        }

def test_magnitude_energy_pruner():
    """测试基于幅值的能量感知剪枝器"""
    print("Testing MagnitudeEnergyPruner...")
    
    # 创建简单模型
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    )
    
    # ✅ 修改为更极端的小值，确保与随机值区分
    model.layers['0'].weight.data = np.random.randn(10, 20) * 0.1
    model.layers['0'].weight.data[:, :4] = 1e-8  # 设为非零但极小值
    
    model.layers['2'].weight.data = np.random.randn(20, 5) * 0.1
    model.layers['2'].weight.data[:3, :] = 1e-8
    
    # 记录剪枝前的参数数量
    initial_params = sum(np.prod(p.shape) for p in model.parameters())
    
    # 创建剪枝器
    pruner = MagnitudeEnergyPruner(sparsity_target=0.3, energy_aware=False)
    mock_monitor = MockEnergyMonitor()
    
    # 执行剪枝
    masks = pruner.prune_model(model, mock_monitor)
    
    # 检查掩码
    assert len(masks) > 0, "Should generate pruning masks"
    
    # 检查参数是否被正确剪枝
    for param_name, mask in masks.items():
        assert mask.shape == model.layers['0'].weight.shape or mask.shape == model.layers['2'].weight.shape, \
            f"Mask shape {mask.shape} should match parameter shape"
        
        # 检查稀疏度
        sparsity = np.sum(mask == 0) / np.prod(mask.shape)
        assert 0.2 <= sparsity <= 0.4, f"Sparsity {sparsity} should be close to target 0.3"

    # 检查剪枝报告
    report = pruner.get_pruning_report()
    assert 'total_parameters_pruned' in report, "Report should contain pruning statistics"
    assert 'average_sparsity' in report, "Report should contain average sparsity"
    
    print("✓ MagnitudeEnergyPruner test passed")

def test_gradient_energy_pruner():
    """测试基于梯度的能量感知剪枝器"""
    print("Testing GradientEnergyPruner...")
    
    model = Sequential(Linear(10, 5))
    
    # # 设置权重和梯度
    # model.layers['0'].weight.data = np.random.randn(10, 5)
    # model.layers['0'].weight.grad = np.ones_like(model.layers['0'].weight.data)
    
    # # 故意设置一些小的权重梯度组合
    # model.layers['0'].weight.data[0, :] = 0.001  # 小的权重
    # model.layers['0'].weight.grad[0, :] = 0.001  # 小的梯度
    
    # 设置权重（随机初始化）
    model.layers['0'].weight.data = np.random.randn(10, 5)
    
    # 设置梯度：大部分为1（正常梯度），第0行为极小值（应该被剪枝）
    model.layers['0'].weight.grad = np.ones((10, 5), dtype=np.float32)
    model.layers['0'].weight.grad[0, :] = 1e-8  # 梯度小 → 重要性低 → 应该被剪枝
    
    pruner = GradientEnergyPruner(sparsity_target=0.2, gradient_power=1.0)
    mock_monitor = MockEnergyMonitor()
    
    # 执行剪枝
    masks = pruner.prune_model(model, mock_monitor)
    
    # 检查掩码
    assert len(masks) == 1, "Should generate one pruning mask"
    
    mask = list(masks.values())[0]
    sparsity = np.sum(mask == 0) / np.prod(mask.shape)
    
    assert 0.15 <= sparsity <= 0.25, f"Sparsity {sparsity} should be close to target 0.2"
    
    print("✓ GradientEnergyPruner test passed")

def test_structured_energy_pruner():
    """测试结构化能量感知剪枝器"""
    print("Testing StructuredEnergyPruner...")
    
    # 创建卷积模型测试通道剪枝
    model = Sequential(
        Conv2d(3, 16, kernel_size=3),
        ReLU(),
        Linear(16 * 30 * 30, 10)  # 假设输入是32x32，卷积后为30x30
    )
    
    # 设置卷积层权重
    model.layers['0'].weight.data = np.random.randn(16, 3, 3, 3)
    
    # 故意设置一些通道的权重很小
    model.layers['0'].weight.data[0:3, :, :, :] = 0.001  # 前3个通道权重很小
    
    pruner = StructuredEnergyPruner(sparsity_target=0.25, structure_type='channel')
    mock_monitor = MockEnergyMonitor()
    
    # 执行剪枝
    masks = pruner.prune_model(model, mock_monitor)
    
    # 检查掩码
    assert len(masks) > 0, "Should generate pruning masks"
    
    # 检查结构化剪枝特性
    conv_mask = masks['layer_0_Conv2d.weight']
    
    # 在通道剪枝中，整个通道应该被剪枝
    # 检查是否有整个通道被置零
    channel_sums = np.sum(conv_mask, axis=(1, 2, 3))
    zero_channels = np.sum(channel_sums == 0)
    
    assert zero_channels > 0, "Should prune entire channels in structured pruning"
    
    print("✓ StructuredEnergyPruner test passed")

def test_progressive_energy_pruner():
    """测试渐进式能量感知剪枝器"""
    print("Testing ProgressiveEnergyPruner...")
    
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    )
    
    # 设置权重
    model.layers['0'].weight.data = np.random.randn(10, 20)
    model.layers['2'].weight.data = np.random.randn(20, 5)
    
    progressive_pruner = ProgressiveEnergyPruner(
        initial_sparsity=0.1,
        final_sparsity=0.6,
        steps=5
    )
    
    mock_monitor = MockEnergyMonitor()
    
    # 执行渐进式剪枝
    for step in range(5):
        masks = progressive_pruner.prune_step(model, mock_monitor)
        
        # 检查进度报告
        progress = progressive_pruner.get_progress_report()
        assert progress['current_step'] == step + 1, f"Current step should be {step + 1}"
        
        if masks:
            # 计算总体稀疏度
            total_pruned = 0
            total_params = 0
            for mask in masks.values():
                total_pruned += np.sum(mask == 0)
                total_params += np.prod(mask.shape)
            
            sparsity = total_pruned / total_params
            expected_sparsity = 0.1 + (0.6 - 0.1) * (step / 4)  # 线性递增
            
            assert abs(sparsity - expected_sparsity) < 0.15, \
                f"Sparsity {sparsity} should be close to expected {expected_sparsity}"
    
    # 检查最终状态
    final_progress = progressive_pruner.get_progress_report()
    assert final_progress['completed'], "Progressive pruning should be completed"
    
    print("✓ ProgressiveEnergyPruner test passed")

def test_energy_importance_calculation():
    """测试能量重要性计算"""
    print("Testing energy importance calculation...")
    
    pruner = MagnitudeEnergyPruner(sparsity_target=0.3, energy_aware=True)
    mock_monitor = MockEnergyMonitor()
    
    # 创建测试参数
    class MockParam:
        def __init__(self, name, data):
            self.name = name
            self.data = data
    
    # 测试不同能量层的调整
    low_energy_param = MockParam("layer_0_Linear.weight", np.ones((10, 20)))
    high_energy_param = MockParam("conv_layer.weight", np.ones((16, 3, 3, 3)))
    
    low_importance = pruner._compute_energy_importance(
        low_energy_param.name, low_energy_param, mock_monitor
    )
    high_importance = pruner._compute_energy_importance(
        high_energy_param.name, high_energy_param, mock_monitor
    )
    
    # 高能量层应该有更高的重要性（更少剪枝）
    assert high_importance > low_importance, \
        f"High energy layer should have higher importance: {high_importance} vs {low_importance}"
    
    # 检查稀疏度调整
    base_sparsity = 0.3
    adjusted_low = pruner._adjust_sparsity_by_energy(base_sparsity, low_importance)
    adjusted_high = pruner._adjust_sparsity_by_energy(base_sparsity, high_importance)
    
    # 高重要性应该导致更低的剪枝率
    assert adjusted_high < adjusted_low, \
        f"High importance should result in lower sparsity: {adjusted_high} vs {adjusted_low}"
    
    print("✓ Energy importance calculation test passed")

def test_pruning_edge_cases():
    """测试剪枝边界情况"""
    print("Testing pruning edge cases...")
    
    # 测试空模型
    empty_model = Sequential()
    pruner = MagnitudeEnergyPruner(sparsity_target=0.3)
    
    masks = pruner.prune_model(empty_model)
    assert len(masks) == 0, "Should handle empty model gracefully"
    
    # 测试全零权重
    zero_model = Sequential(Linear(5, 3))
    zero_model.layers[0].weight.data = np.zeros((5, 3))
    
    masks = pruner.prune_model(zero_model)
    assert len(masks) == 1, "Should handle zero weights"
    
    zero_mask = list(masks.values())[0]
    sparsity = np.sum(zero_mask == 0) / np.prod(zero_mask.shape)
    assert sparsity == 1.0, "Zero weights should result in 100% sparsity"
    
    # 测试无能量监控器的情况
    normal_model = Sequential(Linear(10, 5))
    normal_model.layers[0].weight.data = np.random.randn(10, 5)
    
    masks_no_monitor = pruner.prune_model(normal_model)  # 不传递monitor
    assert len(masks_no_monitor) > 0, "Should work without energy monitor"
    
    print("✓ Pruning edge cases test passed")

def run_all_pruning_tests():
    """运行所有pruning测试"""
    print("Running Energy Pruning tests...")
    print("=" * 50)
    
    test_magnitude_energy_pruner()
    test_gradient_energy_pruner()
    test_structured_energy_pruner()
    test_progressive_energy_pruner()
    test_energy_importance_calculation()
    test_pruning_edge_cases()
    
    print("=" * 50)
    print("All Energy Pruning tests passed! ✅")

if __name__ == "__main__":
    run_all_pruning_tests()