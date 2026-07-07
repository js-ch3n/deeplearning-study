import argparse

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms

from models.cifar10_model import Cifar10Network


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(checkpoint_path, device):
    cfg = load_config()
    model = Cifar10Network().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(image_path, checkpoint_path="checkpoints/best.pth",
            config_path="config.yaml", top_k=5):
    cfg = load_config(config_path)
    device = resolve_device(cfg["device"])

    model = load_model(checkpoint_path, device)
    input_tensor = preprocess(image_path).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for i in range(top_k):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        results.append({
            "rank": i + 1,
            "class": CIFAR10_CLASSES[idx],
            "confidence": round(prob, 4),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 image prediction")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint (default: checkpoints/best.pth)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Show top K predictions (default: 5)")
    args = parser.parse_args()

    results = predict(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        top_k=args.top_k,
    )

    print(f"\n{args.image}")
    print("-" * 40)
    for r in results:
        print(f"  #{r['rank']}  {r['class']:<12}  {r['confidence']:.2%}")
    print()


if __name__ == "__main__":
    main()
