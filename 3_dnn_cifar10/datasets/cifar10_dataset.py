from torchvision import datasets, transforms
from torch.utils.data import DataLoader

datasets.CIFAR10.mirrors = ["https://mirrors.ustc.edu.cn/cifar10/"]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# batch_size = 64

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=batch_size,
#     shuffle=False
# )

# for x, y in train_loader:
#     print(x.shape, y.shape)

# print("训练集数量：", len(train_dataset))
# print("测试集数量：", len(test_dataset))
