import argparse

import torch
import yaml

from data.imdb_dataset import Vocab, load_imdb_data, get_loaders
from models.transformer_model import TransformerClassifier


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_vocab(data_dir="./data/aclImdb", vocab_size=25000):
    train_texts, _, _, _ = load_imdb_data(data_dir)
    vocab = Vocab(max_size=vocab_size)
    vocab.build(train_texts)
    return vocab


def load_model(checkpoint_path, vocab_size, device):
    model = TransformerClassifier(vocab_size=vocab_size).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def predict(text, model, vocab, device, max_len=256):
    encoded = vocab.encode(text, max_len=max_len)
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    confidence = probs[pred_idx].item()
    label = "pos" if pred_idx == 1 else "neg"

    return {
        "label": label,
        "class_id": pred_idx,
        "confidence": round(confidence, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="IMDB 情感预测")
    parser.add_argument("--text", type=str, help="待预测文本（不传则进入交互模式）")
    parser.add_argument("--evaluate", action="store_true", help="在测试集上评估")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    print(f"Using device: {device}")

    data_cfg = cfg.get("data", {})
    data_folder = data_cfg.get("folder", "./data/aclImdb") if isinstance(data_cfg, dict) else str(data_cfg)
    model_cfg = cfg.get("model", {})
    max_len = model_cfg.get("max_len", 256)
    vocab_size = model_cfg.get("vocab_size", 25000)

    vocab = load_vocab(data_folder, vocab_size=vocab_size)
    model = load_model(args.checkpoint, vocab_size=len(vocab), device=device)
    print(f"Vocab size: {len(vocab)}, Model loaded: {args.checkpoint}\n")

    if args.evaluate:
        from train import test_from_file
        test_from_file(args.checkpoint, args.config)

    elif args.text:
        result = predict(args.text, model, vocab, device, max_len=max_len)
        print(f"Input: {args.text}")
        print(f"  Predicted: {result['label']}  (id={result['class_id']})  confidence={result['confidence']:.2%}")

    else:
        print("交互模式（输入 'quit' 退出）\n")
        while True:
            try:
                text = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip().lower() in ("quit", "exit", "q"):
                break
            if not text.strip():
                continue
            result = predict(text, model, vocab, device, max_len=max_len)
            print(f"  [{result['label']}]  confidence: {result['confidence']:.2%}\n")


if __name__ == "__main__":
    main()
