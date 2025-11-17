import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy.nas import EnergyAwareNAS, EvolutionaryEnergyNAS
from nn import Sequential, Linear, ReLU
from core.tensor import Tensor

def mock_train_function(model, epochs=1):
    """模拟训练函数，返回准确率"""
    # 在实际测试中，这里会进行真实的训练
    # 这里我们返回一个随机准确率用于测试
    return np.random.uniform(0.5, 0.9)

def mock_energy_calculator(model, input_shape):
    """模拟能量计算器"""
    total_params = sum(np.prod(p.shape) for p in model.parameters())
    total_flops = 1000000  # 简化估计
    
    params_efficiency = 1.0 / (1.0 + total_params / 1e6)
    flops_efficiency = 1.0 / (1.0 + total_flops / 1e9)
    energy_efficiency = 0.7 * flops_efficiency + 0.3 * params_efficiency
    
    return {
        'efficiency': energy_efficiency,
        'total_params': total_params,
        'total_flops': total_flops,
        'params_efficiency': params_efficiency,
        'flops_efficiency': flops_efficiency
    }

def test_energy_aware_nas_basic():
    """测试基础能量感知NAS"""
    print("Testing EnergyAwareNAS basic functionality...")
    
    # 定义搜索空间
    search_space = {
        'layer_types': ['Linear', 'ReLU'],
        'hidden_sizes': [64, 128, 256, 512],
        'max_layers': 8
    }
    
    # 创建NAS实例
    nas = EnergyAwareNAS(
        search_space=search_space,
        energy_weight=0.3,
        accuracy_weight=0.7,
        latency_weight=0.0
    )
    
    # 测试架构采样
    architecture = nas._sample_architecture()
    assert 'layers' in architecture, "Architecture should contain layers"
    assert len(architecture['layers']) >= 3, "Should have at least 3 layers (input, hidden, output)"
    
    # 测试架构评估
    input_shape = (32, 784)  # MNIST-like input
    score, metrics = nas.evaluate_architecture(
        architecture, mock_train_function, input_shape, mock_energy_calculator
    )
    
    assert isinstance(score, float), "Score should be a float"
    assert 'accuracy' in metrics, "Metrics should contain accuracy"
    assert 'energy_efficiency' in metrics, "Metrics should contain energy efficiency"
    
    print("✓ EnergyAwareNAS basic test passed")

def test_energy_aware_nas_random_search():
    """测试随机搜索NAS"""
    print("Testing EnergyAwareNAS random search...")
    
    search_space = {
        'layer_types': ['Linear', 'ReLU'],
        'hidden_sizes': [64, 128, 256],
        'max_layers': 6
    }
    
    nas = EnergyAwareNAS(
        search_space=search_space,
        energy_weight=0.4,  # 更高的能量权重
        accuracy_weight=0.6
    )
    
    # 运行随机搜索
    results = nas.random_search(
        num_iterations=5,  # 少量迭代用于测试
        train_func=mock_train_function,
        input_shape=(32, 784),
        energy_calculator=mock_energy_calculator
    )
    
    # 检查结果
    assert 'best_architecture' in results, "Results should contain best architecture"
    assert 'best_score' in results, "Results should contain best score"
    assert 'all_architectures' in results, "Results should contain all architectures"
    
    best_score = results['best_score']
    all_scores = [arch['score'] for arch in results['all_architectures']]
    
    assert best_score == max(all_scores), "Best score should be the maximum of all scores"
    assert len(results['all_architectures']) == 5, "Should evaluate exactly 5 architectures"
    
    # 检查搜索报告
    report = nas.get_search_report()
    assert 'total_evaluations' in report, "Report should contain total evaluations"
    assert 'best_score' in report, "Report should contain best score"
    assert 'average_score' in report, "Report should contain average score"
    
    print("✓ EnergyAwareNAS random search test passed")

def test_evolutionary_energy_nas():
    """测试进化能量感知NAS"""
    print("Testing EvolutionaryEnergyNAS...")
    
    search_space = {
        'layer_types': ['Linear', 'ReLU'],
        'hidden_sizes': [64, 128, 256],
        'max_layers': 6
    }
    
    evolutionary_nas = EvolutionaryEnergyNAS(
        search_space=search_space,
        population_size=8,  # 小种群用于测试
        mutation_rate=0.2,
        crossover_rate=0.8,
        energy_weight=0.3,
        accuracy_weight=0.7
    )
    
    # 初始化种群
    evolutionary_nas.initialize_population()
    assert len(evolutionary_nas.population) == 8, "Should initialize population with correct size"
    
    # 测试进化操作
    # 创建模拟评估种群
    mock_evaluated = []
    for i, arch in enumerate(evolutionary_nas.population):
        mock_evaluated.append((arch, i * 0.1, {}))  # 分数递增
    
    # 测试选择
    selected = evolutionary_nas._select(mock_evaluated, selection_ratio=0.5)
    assert len(selected) == len(evolutionary_nas.population), "Selection should maintain population size"
    
    # 测试交叉
    if len(selected) >= 2:
        parent1, parent2 = selected[0], selected[1]
        child = evolutionary_nas._crossover(parent1, parent2)
        assert 'layers' in child, "Child should have layers"
    
    # 测试变异
    test_individual = {'layers': [{'type': 'Linear', 'in_features': 784, 'out_features': 128}]}
    mutated = evolutionary_nas._mutate(test_individual)
    assert 'layers' in mutated, "Mutated individual should have layers"
    
    print("✓ EvolutionaryEnergyNAS test passed")

def test_energy_efficiency_evaluation():
    """测试能量效率评估"""
    print("Testing energy efficiency evaluation...")
    
    nas = EnergyAwareNAS(search_space={})
    
    # 创建测试模型
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10)
    )
    
    input_shape = (32, 784)
    
    # 测试能量效率评估
    energy_efficiency, metrics = nas._evaluate_energy_efficiency(
        model, input_shape, mock_energy_calculator
    )
    
    assert isinstance(energy_efficiency, float), "Energy efficiency should be a float"
    assert 0 <= energy_efficiency <= 1, "Energy efficiency should be between 0 and 1"
    assert 'total_params' in metrics, "Metrics should contain total parameters"
    assert 'total_flops' in metrics, "Metrics should contain total FLOPs"
    
    # 测试无能量计算器的情况
    energy_efficiency_default, metrics_default = nas._evaluate_energy_efficiency(
        model, input_shape, None  # 不提供能量计算器
    )
    
    assert isinstance(energy_efficiency_default, float), "Default energy efficiency should be a float"
    assert 0 <= energy_efficiency_default <= 1, "Default energy efficiency should be between 0 and 1"
    
    print("✓ Energy efficiency evaluation test passed")

def test_latency_evaluation():
    """测试延迟评估"""
    print("Testing latency evaluation...")
    
    nas = EnergyAwareNAS(search_space={})
    
    # 创建测试模型
    small_model = Sequential(Linear(100, 50))
    large_model = Sequential(
        Linear(1000, 500),
        ReLU(),
        Linear(500, 100),
        ReLU(),
        Linear(100, 10)
    )
    
    input_shape = (32, 100)
    
    # 测试延迟评估
    small_latency = nas._evaluate_latency(small_model, input_shape)
    large_latency = nas._evaluate_latency(large_model, input_shape)
    
    assert isinstance(small_latency, float), "Latency score should be a float"
    assert 0 <= small_latency <= 1, "Latency score should be between 0 and 1"
    assert 0 <= large_latency <= 1, "Latency score should be between 0 and 1"
    
    # 小模型应该有更高的延迟分数（更好）
    assert small_latency > large_latency, \
        f"Small model should have better latency score: {small_latency} vs {large_latency}"
    
    print("✓ Latency evaluation test passed")

def test_nas_weight_configurations():
    """测试NAS权重配置"""
    print("Testing NAS weight configurations...")
    
    search_space = {'layer_types': ['Linear'], 'hidden_sizes': [128], 'max_layers': 4}
    
    # 测试不同权重配置
    test_cases = [
        {'energy_weight': 0.1, 'accuracy_weight': 0.9},  # 重视准确率
        {'energy_weight': 0.5, 'accuracy_weight': 0.5},  # 平衡
        {'energy_weight': 0.9, 'accuracy_weight': 0.1},  # 重视能量效率
    ]
    
    for weights in test_cases:
        nas = EnergyAwareNAS(
            search_space=search_space,
            energy_weight=weights['energy_weight'],
            accuracy_weight=weights['accuracy_weight']
        )
        
        architecture = nas._sample_architecture()
        score, metrics = nas.evaluate_architecture(
            architecture, mock_train_function, (32, 784), mock_energy_calculator
        )
        
        # 验证分数计算
        expected_score = (
            weights['accuracy_weight'] * metrics['accuracy'] +
            weights['energy_weight'] * metrics['energy_efficiency']
        )
        
        assert abs(score - expected_score) < 1e-6, \
            f"Score calculation incorrect for weights {weights}: {score} vs {expected_score}"
    
    print("✓ NAS weight configurations test passed")

def run_all_nas_tests():
    """运行所有NAS测试"""
    print("Running Energy NAS tests...")
    print("=" * 50)
    
    test_energy_aware_nas_basic()
    test_energy_aware_nas_random_search()
    test_evolutionary_energy_nas()
    test_energy_efficiency_evaluation()
    test_latency_evaluation()
    test_nas_weight_configurations()
    
    print("=" * 50)
    print("All Energy NAS tests passed! ✅")

if __name__ == "__main__":
    run_all_nas_tests()