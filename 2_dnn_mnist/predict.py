import torch
from models.mnist_model import MnistNetwork
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MnistNetwork().to(device)
model.load_state_dict(torch.load("checkpoints/mnist.pth"))


data_folder = "./data"

def load_data(data_folder, batch_size=1):
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

train_loader, test_loader = load_data(data_folder)
x, y = next(iter(test_loader))
x = x.to(device)
y = y.to(device)

with torch.no_grad():
    y_pred = model(x)
    print(y_pred)
    print(y)

    print(111)
