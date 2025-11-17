import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn import Sequential, Linear, ReLU
from core import optim, cross_entropy
from energy import EnergyMonitor, EnergyAwareRegularizer, MagnitudeEnergyPruner

# 创建模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

# 能量监控
energy_monitor = EnergyMonitor(model)
energy_monitor.attach()

# 能量感知正则化
energy_regularizer = EnergyAwareRegularizer(
    energy_coeff=1e-8,
    sparsity_coeff=1e-3
)

# 能量感知剪枝
pruner = MagnitudeEnergyPruner(sparsity_target=0.3)

# 训练循环中使用
def train_with_energy_awareness(model, dataloader, epochs=10):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = cross_entropy()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播（被energy_monitor监控）
            output = model(data)
            
            # 计算损失
            data_loss = criterion(output, target)
            
            # 添加能量感知正则化
            reg_loss = energy_regularizer(model, energy_monitor.activation_stats)
            total_batch_loss = data_loss + reg_loss
            
            # 反向传播
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.data
            
        # 每5个epoch打印能量报告
        if epoch % 5 == 0:
            report = energy_monitor.generate_report()
            print(f"Epoch {epoch}: Energy={report['total_energy']:.2f}, "
                  f"Sparsity={report['average_sparsity']:.4f}")
            
            # 应用剪枝（可选，可以在训练后期应用）
            if epoch == epochs - 1:
                pruner.prune_model(model, energy_monitor)
    
    energy_monitor.detach()
    return model