# Deep Learning Study

A collection of PyTorch-based deep learning exercises, covering models from classic fully-connected networks to Transformers.

## Project Structure

```
deep-learning-study/
├── 1_dnn_iris/          # Iris classification with DNN
├── 2_dnn_mnist/         # Handwritten digit recognition (MNIST) with DNN
├── 3_dnn_cifar10/       # Image classification (CIFAR-10) with DNN
├── 4_transformer_imdb/  # Sentiment analysis (IMDB) with Transformer
├── requirements.txt     # Python dependencies
└── README.md
```

## Modules

| Module | Task | Model | Dataset |
|--------|------|-------|---------|
| `1_dnn_iris` | Flower classification | MLP | Iris |
| `2_dnn_mnist` | Digit recognition | MLP | MNIST |
| `3_dnn_cifar10` | Image classification | MLP | CIFAR-10 |
| `4_transformer_imdb` | Sentiment analysis | Transformer Encoder | IMDB |

## Unified Training Framework

Each module follows a consistent three-layer structure:

```
<module>/
├── datasets/   # Dataset loading & preprocessing
├── models/     # Model definition
└── train.py    # Training, evaluation & plotting entry point
```

Every training pipeline includes:

- Fixed random seed for reproducibility (SEED=42)
- Adam optimizer with StepLR learning-rate scheduling
- Cross-entropy loss
- Per-epoch train/test loss and accuracy logging
- Model checkpoint saved after training
- Matplotlib training/test loss curves

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run any module by entering its directory and executing train.py
cd 1_dnn_iris && python train.py
cd 2_dnn_mnist && python train.py
cd 3_dnn_cifar10 && python train.py
cd 4_transformer_imdb && python train.py
```

## License

MIT
