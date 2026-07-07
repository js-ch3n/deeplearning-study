import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from datasets.iris_dataset import IrisDataset
from models.iris_model import MyNetwork

import matplotlib.pyplot as plt


# ===========================
# 随机种子
# ===========================
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ===========================
# 超参数
# ===========================
BATCH_SIZE = 16
EPOCHS = 2000
LEARNING_RATE = 0.001
MODEL_PATH = "checkpoints/iris.pth"

# ===========================
# 设备
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_path = "data/iris.csv"

def load_data(data_path, batch_size=32):
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    label = df.iloc[:, -1].values

    classes = sorted(set(label))
    label2id = {label: i for i, label in enumerate(classes)}

    y = [label2id[label] for label in label]
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # 划分数据集train test
    train_len = int(len(X) * 0.8)

    indices = torch.randperm(
        len(X),
        generator=torch.Generator().manual_seed(42)
    )

    train_idx = indices[:train_len]
    test_idx = indices[train_len:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    mean_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0)

    # 标准化
    X_train = (X_train - mean_train) / std_train
    X_test = (X_test - mean_train) / std_train

    # transform = transforms.Compose([
    #     transforms.Normalize(mean_train, std_train)  # 标准化（均值和方差）
    # ])

    dataset_train = IrisDataset(X_train, y_train)
    dataset_test  = IrisDataset(X_test, y_test)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_one_epoch(model, train_loader, test_loader,
                    criterion, optimizer, scheduler, epoch, epochs,
                    train_loss_list, test_loss_list, test_acc_list):
    model.train()

    total_loss = 0.0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        output = model(x)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    scheduler.step()

    # 测试
    test_loss, test_acc = test(model, test_loader)

    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Test Loss: {test_loss:.4f} "
        f"Accuracy: {test_acc:.4f}"
    )

def plot_losses(train_loss_list, test_loss_list):
    plt.figure(figsize=(8, 5))

    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)

    plt.show()

def train():
    model = MyNetwork().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=700,   # 每200个epoch衰减一次
        gamma=0.8       # 学习率变为原来的0.3倍
    )

    train_loader, test_loader = load_data(data_path, batch_size=BATCH_SIZE)

    # 用于绘图
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []

    # 开始训练
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, test_loader, criterion,
                        optimizer, scheduler, epoch, EPOCHS,
                        train_loss_list, test_loss_list, test_acc_list)

    # 保存模型
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print("训练完成，模型已保存。")

    # 绘制Loss曲线
    plot_losses(train_loss_list, test_loss_list)


def test(model=None, test_loader=None, model_path=MODEL_PATH):
    if model is None:
        model = MyNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if test_loader is None:
        _, test_loader = load_data(data_path, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            output = model(x)

            loss = criterion(output, y)
            total_loss += loss.item()

            pred = torch.argmax(output, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return avg_loss, accuracy


if __name__ == "__main__":
    train()
    # test()