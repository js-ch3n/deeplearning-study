# 深度学习练习项目

基于 PyTorch 的深度学习练习集合，涵盖从经典全连接网络到 Transformer 的多种模型实现。

## 项目结构

```
deep-learning-study/
├── 1_dnn_iris/          # DNN 实现鸢尾花分类
├── 2_dnn_mnist/         # DNN 实现手写数字识别 (MNIST)
├── 3_dnn_cifar10/       # DNN 实现 CIFAR-10 图像分类
├── 4_transformer_imdb/  # Transformer 实现 IMDB 情感分析
├── requirements.txt     # Python 依赖
└── README.md            # 英文版说明
```

## 各模块说明

| 模块 | 任务 | 模型 | 数据集 |
|------|------|------|--------|
| `1_dnn_iris` | 鸢尾花分类 | MLP | Iris |
| `2_dnn_mnist` | 手写数字识别 | MLP | MNIST |
| `3_dnn_cifar10` | 图像分类 | MLP | CIFAR-10 |
| `4_transformer_imdb` | 情感分析 | Transformer Encoder | IMDB |

## 统一训练框架

每个模块采用一致的三层结构：

```
<module>/
├── datasets/   # 数据集加载与预处理
├── models/     # 模型定义
└── train.py    # 训练、评估、绘图入口
```

训练流程统一包含：

- 固定随机种子（SEED=42）保证可复现
- Adam 优化器 + StepLR 学习率调度
- 交叉熵损失
- 每 epoch 输出 train/test loss 与 accuracy
- 训练结束保存模型 checkpoint
- 绘制训练/测试 loss 曲线

## 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 进入任意模块目录，运行训练
cd 1_dnn_iris && python train.py
cd 2_dnn_mnist && python train.py
cd 3_dnn_cifar10 && python train.py
cd 4_transformer_imdb && python train.py
```

## License

MIT
