import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


from models.mnist_model import MnistNetwork


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
BATCH_SIZE = 256
EPOCHS = 60
LEARNING_RATE = 0.001
MODEL_PATH = "./checkpoints/mnist.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_folder = "./data"

def load_data(data_folder, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),                    # 转为Tensor，范围[0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST均值和方差）
    ])

    train_dataset = datasets.MNIST(
        root=data_folder,      # 数据保存目录
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_folder,
        train=False,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

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
        f"Test Accuracy: {test_acc:.4f}"
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
    model = MnistNetwork().to(device)

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

    train_loader, test_loader = load_data(data_folder, batch_size=BATCH_SIZE)

    # 用于绘图  
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []

    # 开始训练
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, test_loader,
                        criterion, optimizer, scheduler, epoch, EPOCHS,
                        train_loss_list, test_loss_list, test_acc_list)

    # 保存模型
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print("训练完成，模型已保存。")

    # 绘制Loss曲线
    plot_losses(train_loss_list, test_loss_list)


def test(model=None, test_loader=None, model_path=MODEL_PATH):
    if model is None:
        model = MnistNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if test_loader is None:
        _, test_loader = load_data(data_folder, batch_size=BATCH_SIZE)

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