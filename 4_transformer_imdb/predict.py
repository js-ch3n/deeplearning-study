import argparse

import torch
import yaml

from datasets.imdb_dataset import Vocab, load_imdb_data, get_loaders
from models.transformer_model import TransformerClassifier


def load_vocab(data_dir="./data/aclImdb"):
    train_texts, train_labels, _, _ = load_imdb_data(data_dir)
    vocab = Vocab(max_size=25000)
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

    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="IMDB 情感预测")
    parser.add_argument("--text", type=str, help="待预测文本（不传则进入交互模式）")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/transformer_imdb.pth")
    parser.add_argument("--data", type=str, default="./data/aclImdb")
    parser.add_argument("--max-len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = load_vocab(args.data)
    model = load_model(args.checkpoint, vocab_size=len(vocab), device=device)
    print(f"Vocab size: {len(vocab)}, Model loaded: {args.checkpoint}\n")

    if args.text:
        label, conf = predict(args.text, model, vocab, device, max_len=args.max_len)
        print(f"  [{label}]  confidence: {conf:.2%}")
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
            label, conf = predict(text, model, vocab, device, max_len=args.max_len)
            print(f"  [{label}]  confidence: {conf:.2%}\n")


if __name__ == "__main__":
    main()
