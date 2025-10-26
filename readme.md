# 项目目的

## 这个项目主要是手搓一个全连接神经网咯，CNN，LSTM，transformer等主流网络

# 之后会从底层搭建强化学习算法等机器学习和深度学习算法

# 项目结构

mytorch/                 # 根目录
├── core/                # 核心算子（手写梯度）
│   ├── __init__.py
│   ├── tensor.py        # 简易张量 + 自动微分
│   ├── functional.py    # 激活/损失/度量
│   └── optim.py         # 手写 SGD、Momentum、Adam
├── nn/                  # 网络积木
│   ├── __init__.py
│   ├── layer.py         # Linear、Dropout、BatchNorm
│   ├── model.py         # 可扩展的 Sequential / Module
│   └── init.py          # Xavier、He、Normal 初始化
├── energy/              # 你预留的“吸能”扩展区
│   ├── __init__.py
│   ├── regularize.py    # 能量正则项（L2、L1、神经元功耗）
│   └── monitor.py       # 前向能量审计、稀疏度计算
├── examples/
│   └── mnist_fc.py      # 端到端示例：MNIST 全连接
├── data/
│   └── mnist.pkl        # 运行脚本自动下载
└── tests/               # 单元测试（可选）
    └── test_tensor.py


mytorch/
├── core/                          # 手写自动微分引擎
│   ├── __init__.py               ← 导出 Tensor, functional, optim
│   ├── tensor.py                 ← 阶段 0：最小可微张量 + 链式梯度
│   ├── functional.py             ← 阶段 0：relu/sigmoid/cross_entropy
│   └── optim.py                  ← 阶段 0：SGD & Momentum
├── nn/                            # 网络积木（无依赖注入，可插拔）
│   ├── __init__.py               ← 导出 Linear / Sequential / ...
│   ├── layer.py                  ← 阶段 0：Linear、Dropout
│   ├── model.py                  ← 阶段 0：Sequential 容器
│   └── init.py                   ← 阶段 0：Xavier/He 初始化
├── energy/                        # 吸能试验田（今天可空）
│   ├── __init__.py
│   ├── regularize.py             ← 阶段 0：空或 L2 正则
│   └── monitor.py                ← 阶段 0：空
├── examples/
│   └── mnist_fc.py               ← 阶段 0：端到端训练脚本
├── data/                          # 数据集缓存
│   └── mnist.pkl                 ← 阶段 0：自动下载
└── tests/
    └── test_tensor.py            ← 阶段 0：断言梯度



core/functional.py          ← 追加 conv2d, max_pool2d, col2im 反向
nn/layer.py                 ← 追加 Conv2d, MaxPool2d, Flatten
examples/mnist_cnn.py       ← 新脚本：CNN 版
tests/test_layer.py         ← 断言 conv 梯度

core/functional.py          ← 追加 lstm_cell, softmax, matmul_mask
nn/layer.py                 ← 追加 LSTM, MultiHeadAttention, TransformerBlock
examples/imdb_lstm.py       ← LSTM 文本分类
examples/toy_transformer.py ← 单头 Attention 加和实验
tests/test_rnn.py           ← 梯度/数值双重检查


# core
| 文件              | 一句话作用                                    | 阶段    |
| --------------- | ---------------------------------------- | ----- |
| `tensor.py`     | 手写可微张量 + 链式梯度（支持广播/stride预留）             | 0     |
| `functional.py` | 所有算子前向+反向：relu、conv2d、lstm\_cell、softmax | 0→1→2 |
| `optim.py`      | SGD、Momentum、AdaGrad、Adam                | 0→1   |

# nn
| 文件         | 一句话作用                                      | 阶段    |
| ---------- | ------------------------------------------ | ----- |
| `layer.py` | 可插拔层：Linear、Conv2d、LSTM、MultiHeadAttention | 0→1→2 |
| `model.py` | Sequential、Module（容留参数树）                   | 0     |
| `init.py`  | Xavier、He、Normal、Uniform                   | 0     |

# energy
| 文件              | 一句话作用                   | 阶段   |
| --------------- | ----------------------- | ---- |
| `regularize.py` | 能量正则：FLOPCount、L2、神经元功耗 | 1 开始 |
| `monitor.py`    | 前向审计、稀疏度、碳排放估算          | 1 开始 |

# example
| 文件                   | 一句话作用             | 阶段 |
| -------------------- | ----------------- | -- |
| `mnist_fc.py`        | 全连接 baseline      | 0  |
| `mnist_cnn.py`       | CNN 精度对比          | 1  |
| `imdb_lstm.py`       | LSTM 文本分类         | 2  |
| `toy_transformer.py` | 单头 Attention 加和实验 | 2  |



# test
| 文件               | 一句话作用       | 阶段  |
| ---------------- | ----------- | --- |
| `test_tensor.py` | 梯度数值校验      | 0   |
| `test_layer.py`  | conv/rnn 梯度 | 1→2 |

| Week | 目标                 | 交付物                                  | 是否破坏老代码 |
| ---- | ------------------ | ------------------------------------ | ------- |
| W1   | 全连通跑通MNIST         | 阶段0全部文件                              | ❌       |
| W2   | 手写conv2d+im2col    | functional.py+layer.py+mnist\_cnn.py | ❌       |
| W3   | 能量正则FLOPCount      | energy/regularize.py                 | ❌       |
| W4   | LSTMCell+IMDB      | functional.py+lstm层+imdb\_lstm.py    | ❌       |
| W5   | MultiHeadAttention | layer.py+toy\_transformer.py         | ❌       |
| W6   | 并行训练卡加速            | 可选：cpp\_extension写CUDA               | ❌       |



# 为什么每个目录下面都有__init__.py文件？


## 1. 最小作用是空文件

## 2.可以进行import操作

```
import ML.nn.layer
from ML.core.tensor import Tensor
```

## 3.常见附加作用
| 功能         | 示例代码                                                     | 效果                                             |
| ---------- | -------------------------------------------------------- | ---------------------------------------------- |
| 1. 控制公开接口  | `__all__ = ['Tensor', 'SGD']`                            | `from ML.core import *` 只导入列出的名字          |
| 2. 简化深层导入  | `from .tensor import Tensor`<br>`from .optim import SGD` | 用户只需<br>`from ML.core import Tensor, SGD` |
| 3. 包级初始化逻辑 | `print('ML 0.1.0 ready')`                           | 首次导入包时执行一次                                     |
| 4. 兼容命名空间  | `import sys`<br>`sys.path.append(...)`                   | 动态把子目录加入搜索路径                                   |
