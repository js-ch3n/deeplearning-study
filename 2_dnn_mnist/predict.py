import argparse

import torch
import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.mnist_model import MnistNetwork


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(checkpoint_path, device):
    model = MnistNetwork().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_data(data_folder, batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(
        root=data_folder,
        train=False,
        download=False,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def evaluate(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            pred = torch.argmax(output, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            for i in range(len(y)):
                per_class_total[y[i]] += 1
                if pred[i] == y[i]:
                    per_class_correct[y[i]] += 1

    accuracy = correct / total
    print(f"\n==> Overall Test Accuracy: {accuracy:.2%}\n")
    print("Per-class accuracy:")
    for c in range(10):
        if per_class_total[c] > 0:
            class_acc = per_class_correct[c] / per_class_total[c]
            print(f"  {c}: {class_acc:.2%}  ({per_class_correct[c]}/{per_class_total[c]})")

    return accuracy


def predict_single(model, data_loader, device, indices=None):
    model.eval()

    all_images = []
    all_labels = []

    for x, y in data_loader:
        all_images.append(x)
        all_labels.append(y)

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if indices is None:
        indices = list(range(min(10, len(all_images))))

    print("\nSample predictions:")
    print("-" * 50)

    with torch.no_grad():
        for idx in indices:
            img = all_images[idx:idx + 1].to(device)
            label = all_labels[idx].item()

            output = model(img)
            prob = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(prob).item()
            confidence = prob[pred].item()

            correct = "✓" if pred == label else "✗"
            print(f"  [{idx}]  pred={pred}  true={label}  {correct}  conf={confidence:.2%}")

    print()


def main():
    parser = argparse.ArgumentParser(description="MNIST digit prediction")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on full test set")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data folder (default from config)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of sample predictions to display (default 10)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    model = load_model(args.checkpoint, device)

    data_folder = args.data or (
        cfg["data"].get("folder", "./data") if isinstance(cfg["data"], dict) else str(cfg["data"])
    )
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data folder: {data_folder}")

    test_loader = load_data(data_folder, batch_size=1)

    if args.evaluate:
        evaluate(model, test_loader, device)
    else:
        predict_single(model, test_loader, device, indices=list(range(args.samples)))


if __name__ == "__main__":
    main()
