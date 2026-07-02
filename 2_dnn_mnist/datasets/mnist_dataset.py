from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor(),                    # 转为Tensor，范围[0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST均值和方差）
])

train_dataset = datasets.MNIST(
    root="./data",      # 数据保存目录
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
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