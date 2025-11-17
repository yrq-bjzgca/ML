# 项目目的

## 这个项目主要是手搓一个全连接神经网咯，CNN，LSTM，transformer等主流网络

# 之后会从底层搭建强化学习算法等机器学习和深度学习算法

# 项目结构
```tree
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
```
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


# 使用方法

```python

# 1. 仅安装运行时依赖
pip install -e .

# 2. 连带测试/开发工具一起装
pip install -e ".[test,dev]"

```

自编码器（Autoencoder）：包括去噪自编码器、变分自编码器（VAE）等，用于无监督学习。

生成对抗网络（GAN）：包括DCGAN、WGAN等，用于生成模型。

残差网络（ResNet）：解决深度网络退化问题，可以用于图像分类。

注意力机制（Attention）：除了Transformer中的自注意力，还有各种注意力变体，如通道注意力、空间注意力等。

图神经网络（GNN）：如图卷积网络（GCN）、图注意力网络（GAT）等，用于图结构数据。

强化学习算法：如DQN、Policy Gradients、Actor-Critic等，可以放在强化学习目录下。

归一化层：如BatchNorm、LayerNorm、InstanceNorm、GroupNorm等，这些已经在CNN中常见，但可以扩展到其他网络。

循环神经网络变体：如GRU（Gated Recurrent Unit），是LSTM的简化版。

胶囊网络（Capsule Network）：一种新的图像识别网络，旨在解决CNN的不足。

神经ODE：基于常微分方程的神经网络。


1. ResNet (残差网络)
# 核心：跳跃连接 + 残差块
class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm2d(out_channels)
        
        # 捷径连接
        self.shortcut = Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2d(in_channels, out_channels, 1, stride),
                BatchNorm2d(out_channels)
            )

2. U-Net (编码器-解码器)
class UNet:
    def __init__(self):
        # 编码器 (下采样)
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        # 解码器 (上采样 + 跳跃连接)
        self.dec1 = UpConvBlock(128, 64)

3. VAE (变分自编码器)
class VAE:
    def __init__(self):
        self.encoder = Sequential(...)
        self.fc_mu = Linear(hidden_dim, latent_dim)    # 均值
        self.fc_logvar = Linear(hidden_dim, latent_dim) # 对数方差
        self.decoder = Sequential(...)


4. GAN (生成对抗网络)

class Generator:
    """从噪声生成假数据"""
    def __init__(self):
        self.fc1 = Linear(100, 256)
        self.fc2 = Linear(256, 784)  # MNIST 28x28

class Discriminator:
    """区分真假数据"""
    def __init__(self):
        self.fc1 = Linear(784, 256)
        self.fc2 = Linear(256, 1)

5. Attention-based CNN (SENet, CBAM)
class SEBlock:
    """通道注意力"""
    def __init__(self, channel, reduction=16):
        self.gap = AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = Sequential(
            Linear(channel, channel // reduction),
            Linear(channel // reduction, channel)
        )

6. Lightweight CNN
MobileNet (深度可分离卷积)

ShuffleNet (通道混洗)

SqueezeNet (Fire Module)

7.. AutoML相关
NASCell (神经架构搜索单元)

DARTS (可微分架构搜索)



场景	推荐网络	特点
图像分类	ResNet, EfficientNet	深度/宽度/分辨率缩放
目标检测	YOLO, SSD	单阶段检测器
语义分割	U-Net, DeepLab	编码器-解码器结构
生成模型	VAE, GAN, Diffusion	概率建模/对抗训练
轻量化	MobileNet, ShuffleNet	移动端部署