import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn import Sequential, Linear, ReLU
from core import Tensor
import numpy as np
import pandas as pd
import time
from core import cross_entropy, optim

class MLP(Sequential):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, output_size=10):
        super().__init__(
            Linear(input_size, hidden1),
            ReLU(),
            Linear(hidden1, hidden2),
            ReLU(), 
            Linear(hidden2, output_size)
        )

def load_csv(path, samples=None):
    df = pd.read_csv(path)
    if samples:
        df = df.iloc[:samples]
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    y = df.iloc[:, 0].values.astype(np.int32)
    return X, y

def accuracy(logits, labels):
    preds = np.argmax(logits.data, axis=1)
    return (preds == labels.data).mean()

# 详细的调试函数
def debug_model(model, x_batch, y_batch):
    print("\n=== 模型调试信息 ===")
    
    # 检查输入数据
    print(f"输入数据范围: [{x_batch.data.min():.4f}, {x_batch.data.max():.4f}]")
    print(f"标签: {y_batch.data[:10]}...")
    
    # 逐层前向传播并检查
    x = x_batch
    for i, layer in enumerate(model):
        print(f"\n第{i}层 {type(layer).__name__}:")
        x_prev = x
        x = layer(x)
        print(f"  输入范围: [{x_prev.data.min():.4f}, {x_prev.data.max():.4f}]")
        print(f"  输出范围: [{x.data.min():.4f}, {x.data.max():.4f}]")
        
        # 检查权重
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            print(f"  权重范围: [{w.min():.4f}, {w.max():.4f}], 均值: {w.mean():.4f}")
            if hasattr(layer, 'bias_param') and layer.bias_param is not None:
                b = layer.bias_param.data
                print(f"  偏置范围: [{b.min():.4f}, {b.max():.4f}], 均值: {b.mean():.4f}")
    
    logits = x
    print(f"\n最终logits范围: [{logits.data.min():.4f}, {logits.data.max():.4f}]")
    print(f"Logits样例: {logits.data[0][:5]}...")
    
    return logits

def debug_gradients(model):
    print("\n=== 梯度调试信息 ===")
    total_params = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = np.linalg.norm(param.grad.reshape(-1))
            print(f"{name}: 梯度范数 = {grad_norm:.4f}")
            total_params += 1
        else:
            print(f"{name}: 梯度为None")
    print(f"总参数数量: {total_params}")

# 超参数
BATCH = 64
EPOCHS = 5
LR = 0.01  # 保持你的学习率

X_train, y_train = load_csv('../data/digit-recognizer/train.csv', samples=1000)  # 先用小数据测试

model = MLP()
opt = optim.SGD(model.parameters(), lr=LR)

# steps = min(20, len(X_train) // BATCH)  # 只跑几步测试
steps = 200

print("=== 初始模型状态 ===")
# 检查初始权重
for name, param in model.named_parameters():
    print(f"{name}: [{param.data.min():.4f}, {param.data.max():.4f}], 均值: {param.data.mean():.4f}")

for epoch in range(EPOCHS):
    t0 = time.time()
    loss_sum, acc_sum = 0., 0.
    
    for s in range(steps):
        idx = np.random.randint(0, len(X_train), BATCH)
        x_batch = Tensor(X_train[idx])
        y_batch = Tensor(y_train[idx])
        
        # 第一次迭代时详细调试
        if epoch == 0 and s == 0:
            logits = debug_model(model, x_batch, y_batch)
        else:
            logits = model(x_batch)
            
        loss = cross_entropy(logits, y_batch)
        
        print(f"\nStep {s}, Loss: {loss.data:.4f}")
        
        # 反向传播
        opt.zero_grad()
        loss.backward()
        
        # 调试梯度
        if epoch == 0 and s == 0:
            debug_gradients(model)
        
        opt.step()

        loss_sum += loss.data
        acc_sum += accuracy(logits, y_batch)
        
        # 如果loss太大就提前停止
        if loss.data > 1000:
            print(f"Loss过大，提前停止训练")
            break

    print(f'Epoch {epoch+1}/{EPOCHS}  loss={loss_sum/steps:.4f}  train_acc={acc_sum/steps:.4f}  time={time.time()-t0:.1f}s')
    
    # 如果loss爆炸就停止
    if loss_sum/steps > 1000:
        break