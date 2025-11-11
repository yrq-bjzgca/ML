import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn import Sequential, Linear, ReLU, Conv2d, MaxPool2d, Flatten
from core import Tensor
import numpy as np
import pandas as pd
import time
from core import cross_entropy, optim

# 添加详细的调试函数
def debug_model_parameters(model, prefix=""):
    """打印模型参数的统计信息"""
    print(f"\n{prefix}模型参数统计:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.data is not None:
            print(f"  {name}: shape={param.shape}, mean={np.mean(param.data):.6f}, "
                  f"std={np.std(param.data):.6f}, min={np.min(param.data):.6f}, max={np.max(param.data):.6f}")
            total_params += np.prod(param.shape)
    print(f"总参数数量: {total_params}")

def debug_forward_pass(model, x_batch):
    """调试前向传播的每一层输出"""
    print("\n前向传播调试:")
    
    # 手动执行每一层并检查输出
    x = x_batch
    for i, layer in enumerate(model):
        x_prev = x
        x = layer(x)
        print(f"第{i}层 {type(layer).__name__}:")
        print(f"  输入: shape={x_prev.shape}, range=[{x_prev.data.min():.4f}, {x_prev.data.max():.4f}]")
        print(f"  输出: shape={x.shape}, range=[{x.data.min():.4f}, {x.data.max():.4f}]")
        
        # 检查是否有NaN或Inf
        if np.any(np.isnan(x.data)):
            print(f"  ⚠️ 第{i}层输出包含NaN!")
        if np.any(np.isinf(x.data)):
            print(f"  ⚠️ 第{i}层输出包含Inf!")
        
        if i >= 5:  # 只显示前几层，避免输出过多
            print("  ...")
            break
    
    return x

def debug_gradients(model, prefix=""):
    """打印梯度信息"""
    print(f"\n{prefix}梯度统计:")
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = np.linalg.norm(param.grad.reshape(-1))
            print(f"  {name}: grad_norm={grad_norm:.6f}, "
                  f"mean={np.mean(param.grad):.6f}")
            has_gradients = True
            
            # 检查梯度问题
            if np.any(np.isnan(param.grad)):
                print(f"  ⚠️ {name} 梯度包含NaN!")
            if np.any(np.isinf(param.grad)):
                print(f"  ⚠️ {name} 梯度包含Inf!")
            if grad_norm < 1e-10:
                print(f"  ⚠️ {name} 梯度消失!")
            if grad_norm > 1e5:
                print(f"  ⚠️ {name} 梯度爆炸!")
    
    if not has_gradients:
        print("  没有检测到梯度!")

class CNN(Sequential):
    def __init__(self):
        super().__init__(
            Conv2d(1, 32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            
            Flatten(),
            Linear(64 * 7 * 7, 128),
            ReLU(),
            Linear(128, 10)
        )

# ---------- 数据加载和预处理 ----------
def load_csv(path, samples=None):
    """返回 X(N,784), y(N) 均为 float/int numpy"""
    df = pd.read_csv(path)
    if samples:
        df = df.iloc[:samples]
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    y = df.iloc[:, 0].values.astype(np.int32)
    return X, y

def preprocess_data(X):
    """将数据从(N, 784)转换为(N, 1, 28, 28)"""
    X_reshaped = X.reshape(-1, 1, 28, 28)
    return X_reshaped

# 使用更小的配置进行调试
BATCH = 32
EPOCHS = 5
LR = 0.0001  # 更小的学习率

# 加载少量数据进行快速调试
X_train, y_train = load_csv('../data/digit-recognizer/train.csv', samples=1000)
X_train_cnn = preprocess_data(X_train)

print(f"训练数据形状: {X_train_cnn.shape}")
print(f"标签范围: {np.min(y_train)} 到 {np.max(y_train)}")

# ---------- 训练 ----------
model = CNN()
debug_model_parameters(model, "初始化后")

opt = optim.Adam(model.parameters(), lr=LR)

print("开始调试训练...")

# 先测试一个batch
idx = np.random.randint(0, len(X_train_cnn), BATCH)
x_batch = Tensor(X_train_cnn[idx])
y_batch = Tensor(y_train[idx])

print(f"输入数据统计: mean={np.mean(x_batch.data):.4f}, std={np.std(x_batch.data):.4f}")

# 调试前向传播
with Tensor.no_grad():
    logits = debug_forward_pass(model, x_batch)
    print(f"最终输出统计: mean={np.mean(logits.data):.4f}, std={np.std(logits.data):.4f}")

# 完整的训练步骤
print("\n执行完整训练步骤:")
logits = model(x_batch)
print(f"训练模式输出范围: [{logits.data.min():.4f}, {logits.data.max():.4f}]")

loss = cross_entropy(logits, y_batch)
print(f"损失值: {loss.data}")

# 检查损失计算
print(f"Logits softmax: {np.exp(logits.data) / np.sum(np.exp(logits.data), axis=1, keepdims=True)}")

loss.backward()
debug_gradients(model, "反向传播后")

opt.step()
opt.zero_grad()

print("\n第一次参数更新完成")
debug_model_parameters(model, "第一次更新后")