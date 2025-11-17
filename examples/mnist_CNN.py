import sys
import os

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn import Sequential, Linear, ReLU, Conv2d, MaxPool2d, Flatten
from core import Tensor
import numpy as np
import pandas as pd
import time
from core import cross_entropy, optim

class CNN(Sequential):
    def __init__(self):
        super().__init__(
            Conv2d(1,32,kernel_size=3,padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),

            Conv2d(32,64,kernel_size=3,padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
      
            Flatten(),
            Linear(3136,128),
            ReLU(),
            Linear(128,10)
        )

# ---------- 数据加载和预处理 ----------
def load_csv(path, samples=None):
    """返回 X(N,784), y(N) 均为 float/int numpy"""
    df = pd.read_csv(path)
    if samples:                      # 只取前若干行，方便调试
        df = df.iloc[:samples]
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    y = df.iloc[:, 0].values.astype(np.int32)
    return X, y

def preprocess_data(X):
    """将数据从(N, 784)转换为(N, 1, 28, 28)以适应CNN输入"""
    # 重塑为(N, 1, 28, 28)，其中1是通道数
    X_reshaped = X.reshape(-1, 1, 28, 28)
    return X_reshaped

BATCH = 64
EPOCHS = 10          # 先跑10个epoch看看
LR = 0.001

# 加载数据
X_train, y_train = load_csv('../data/digit-recognizer/train.csv', samples=30_000)

# 预处理数据 - 重塑为CNN需要的形状
X_train_cnn = preprocess_data(X_train)

print(f"训练数据形状: {X_train_cnn.shape}")
print(f"标签形状: {y_train.shape}")


# ---------- 指标 ----------
def accuracy(logits, labels):
    # logits: (B,10) Tensor, labels: (B,) Tensor
    preds = np.argmax(logits.data, axis=1)
    return (preds == labels.data).mean()

# ---------- 训练 ----------
model = CNN()
# print("模型结构:")
# print(model)

# 使用较小的学习率，因为CNN通常需要更小的学习率
opt = optim.Adam(model.parameters(), lr=LR)

# 计算总步数
steps_per_epoch = len(X_train) // BATCH
total_steps = EPOCHS * steps_per_epoch

print(f"开始训练，总步数: {total_steps}")
EPOCHS = 100
steps_per_epoch = 10
for epoch in range(EPOCHS):
    t0 = time.time()
    loss_sum, acc_sum = 0., 0.
    
    for step in range(steps_per_epoch):
        # 随机选择batch
        idx = np.random.randint(0, len(X_train_cnn), BATCH)
        x_batch = Tensor(X_train_cnn[idx])  # (B, 1, 28, 28)
        y_batch = Tensor(y_train[idx])      # (B,)
        
        # 前向传播
        logits = model(x_batch)
        loss = cross_entropy(logits, y_batch)
        
        # 反向传播
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        # 累计统计
        loss_sum += loss.data
        acc_sum += accuracy(logits, y_batch)
        
        # 每100步打印一次进度
        if step % 100 == 0:
            current_step = epoch * steps_per_epoch + step
            progress = current_step / total_steps * 100
            print(f'进度: {progress:.1f}% | Step {step}/{steps_per_epoch} | '
                  f'Loss: {loss.data:.4f} | Acc: {accuracy(logits, y_batch):.4f}')

    # 每个epoch结束后打印统计信息
    avg_loss = loss_sum / steps_per_epoch
    avg_acc = acc_sum / steps_per_epoch
    epoch_time = time.time() - t0
    
    print(f'Epoch {epoch+1}/{EPOCHS} | '
          f'Loss: {avg_loss:.4f} | '
          f'Acc: {avg_acc:.4f} | '
          f'Time: {epoch_time:.1f}s')
    print('-' * 50)

print("训练完成!")

