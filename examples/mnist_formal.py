import sys
import os

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.nn as nn 
import numpy as np
import pandas as pd
import  time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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
EPOCHS = 5          # 先跑 5 个 epoch 看看
LR     = 0.01

X_train, y_train = load_csv('../data/digit-recognizer/train.csv', samples=30_000)  # 3 万行先试试
X_train = torch.from_numpy(X_train).to(device)
y_train = torch.from_numpy(y_train).to(device)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)


# ---------- 指标 ----------
def accuracy(logits, labels):
    # logits: (B,10)  Tensor,  labels: (B,) Tensor
    preds = np.argmax(logits.data, axis=1)
    # print(preds)
    # print(labels.data)
    # print(f"preds={preds},label = {labels.data}")
    return (preds == labels.data).mean()

# ---------- 训练 ----------
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
steps = len(X_train) // BATCH
for epoch in range(EPOCHS):
    t0 = time.time()
    loss_sum, acc_sum = 0., 0.
    for s in range(steps):
        # 切片取 batch
        idx = torch.randint(0, len(X_train), (BATCH, ))
        x_batch = X_train[idx]        # (B,784)
        y_batch = y_train[idx]        # (B,)
        
        # # 前向
        # logits = model(x_batch)
        # # print(logits)
        # loss   = cross_entropy(logits, y_batch)

        # # 反向
        # loss.backward()
        # opt.step()
        # opt.zero_grad()

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch.long())
        # print(loss)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        acc_sum  += (logits.argmax(1)==y_batch).float().mean().item()



    print(f'Epoch {epoch+1}/{EPOCHS}  '
          f'loss={loss_sum/steps:.4f}  '
          f'train_acc={acc_sum/steps:.4f}  '
          f'time={time.time()-t0:.1f}s')
    
    