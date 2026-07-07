import argparse

import torch
import yaml

from models.iris_model import MyNetwork


IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(checkpoint_path, device):
    model = MyNetwork().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def predict(features, model, device):
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
    if features.dim() == 1:
        features = features.unsqueeze(0)

    features = features.to(device)

    with torch.no_grad():
        output = model(features)
        probabilities = torch.softmax(output, dim=1)[0]

    pred_idx = torch.argmax(probabilities).item()
    confidence = probabilities[pred_idx].item()

    return {
        "class": IRIS_CLASSES[pred_idx],
        "class_id": pred_idx,
        "confidence": round(confidence, 4),
        "probabilities": {
            IRIS_CLASSES[i]: round(probabilities[i].item(), 4)
            for i in range(len(IRIS_CLASSES))
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Iris flower prediction")
    parser.add_argument("--features", type=float, nargs=4,
                        help="4 features: sepal_length sepal_width petal_length petal_width")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on the test split of the csv dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    model = load_model(args.checkpoint, device)
    print(f"Using device: {device}")

    if args.evaluate:
        from train import test_from_file
        test_from_file(args.checkpoint, args.config)

    elif args.features:
        result = predict(args.features, model, device)
        print(f"\nInput: {args.features}")
        print(f"  Predicted: {result['class']}  (id={result['class_id']})  confidence={result['confidence']:.2%}")
        print(f"  Probabilities: {result['probabilities']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
