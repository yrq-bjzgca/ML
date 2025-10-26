├── core/                # 核心算子（手写梯度）
│   ├── __init__.py
│   ├── tensor.py        # 简易张量 + 自动微分
│   ├── functional.py    # 激活/损失/度量
│   └── optim.py         # 手写 SGD、Momentum、Adam