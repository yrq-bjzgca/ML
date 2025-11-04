import sys
import os

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn import Sequential, Linear, ReLU
from core import Tensor
import numpy as np
import pandas as pd
import  time
from core import cross_entropy, optim

class MLP(Sequential):
    def __init__(self, input_size = 784,hidden1 = 256,hidden2 =128, output_size = 10):
        super().__init__(
            Linear(input_size,hidden1),
            ReLU(),
            Linear(hidden1,hidden2),
            ReLU(),
            Linear(hidden2,output_size)
        )

# ---------- 数据加载 ----------
def load_csv(path, samples=None):
    """返回 X(N,784), y(N) 均为 float/int  numpy"""
    df = pd.read_csv(path)
    if samples:                      # 只取前若干行，方便调试
        df = df.iloc[:samples]
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    y = df.iloc[:, 0].values.astype(np.int32)
    return X, y

BATCH = 64
EPOCHS = 100          # 先跑 5 个 epoch 看看
LR     = 0.01


X_train, y_train = load_csv('../data/digit-recognizer/train.csv', samples=30_000)  # 3 万行先试试

# ---------- 指标 ----------
def accuracy(logits, labels):
    # logits: (B,10)  Tensor,  labels: (B,) Tensor
    preds = np.argmax(logits.data, axis=1)
    return (preds == labels.data).mean()

# ---------- 训练 ----------
model = MLP()
opt   = optim.SGD(model.parameters(), lr=LR)

steps = 10 
for epoch in range(EPOCHS):
    t0 = time.time()
    loss_sum, acc_sum = 0., 0.
    for s in range(steps):
        # 切片取 batch
        idx = np.random.randint(0, len(X_train), BATCH)
        x_batch = Tensor(X_train[idx])          # (B,784)
        y_batch = Tensor(y_train[idx])          # (B,)
        # print(x_batch.data[0].min(), x_batch.data[0].max(), y_batch.data[:4])
        # 前向
        logits = model(x_batch)
        
        loss   = cross_entropy(logits, y_batch)
        # print(loss)
        # 反向
        loss.backward()

        opt.step()
 
        opt.zero_grad()

        loss_sum += loss.data
        acc_sum  += accuracy(logits, y_batch)

    print(f'Epoch {epoch+1}/{EPOCHS}  '
          f'loss={loss_sum/steps:.4f}  '
          f'train_acc={acc_sum/steps:.4f}  '
          f'time={time.time()-t0:.1f}s')
