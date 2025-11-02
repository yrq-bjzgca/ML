```tree
├── nn/                  # 网络积木
│   ├── __init__.py
│   ├── layer.py         # Linear、Dropout、BatchNorm
│   ├── model.py         # 可扩展的 Sequential / Module
│   └── init.py          # Xavier、He、Normal 初始化
```


# 关于__init__文件的作用

清晰的模块文档：说明这个模块的用途

有组织的导入：按照功能分组导入不同的类和方法

明确的API：使用 __all__ 定义了公开的接口

版本信息：便于模块管理

``` python
# 有__init__
from ML.nn import Linear, Sequential, xavier_uniform_
```

``` python
# 没有__init__的时候
from ML.nn.layer import Linear
from ML.nn.model import Sequential  
from ML.nn.init import xavier_uniform_
```

