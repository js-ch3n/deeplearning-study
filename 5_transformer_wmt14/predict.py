import argparse

import torch
import yaml

from data.wmt14_dataset import get_loaders, Vocab, WMT14Dataset
from models.transformer_seq2seq import TransformerSeq2seq


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_vocab_and_model(checkpoint_path, config_path="config.yaml"):
    cfg = load_config(config_path)
    device = resolve_device(cfg["device"])

    # 重新加载词汇表
    data_cfg = cfg["data"]
    train_loader, val_loader, src_vocab, tgt_vocab = get_loaders(
        lang_pair=data_cfg["lang_pair"],
        batch_size=data_cfg["batch_size"],
        max_len=data_cfg["max_len"],
        vocab_size=data_cfg["vocab_size"],
        num_samples=data_cfg.get("num_samples"),
    )

    model_cfg = cfg["model"]
    max_len = data_cfg["max_len"]

    model = TransformerSeq2seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        max_len=max_len,
        dropout=model_cfg["dropout"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, src_vocab, tgt_vocab, cfg, device


def ids_to_text(ids, vocab):
    tokens = []
    for idx in ids:
        if idx in (vocab.BOS_IDX, vocab.PAD_IDX):
            continue
        if idx == vocab.EOS_IDX:
            break
        tokens.append(vocab.idx2word.get(idx, "<UNK>"))
    return " ".join(tokens)


def translate(text, model, src_vocab, tgt_vocab, device, max_len=64, beam_size=1):
    src_encoded = src_vocab.encode(text, max_len=max_len)
    src_tensor = torch.tensor([src_encoded], dtype=torch.long).to(device)

    output_ids = model.translate(
        src_tensor, src_vocab, tgt_vocab,
        max_len=max_len, beam_size=beam_size,
    )
    translation = ids_to_text(output_ids, tgt_vocab)
    return translation, output_ids


def evaluate_dataset(checkpoint_path, config_path="config.yaml", num_samples=100):
    """在验证集前 N 条上展示翻译结果"""
    cfg = load_config(config_path)
    device = resolve_device(cfg["device"])

    data_cfg = cfg["data"]
    train_loader, val_loader, src_vocab, tgt_vocab = get_loaders(
        lang_pair=data_cfg["lang_pair"],
        batch_size=data_cfg["batch_size"],
        max_len=data_cfg["max_len"],
        vocab_size=data_cfg["vocab_size"],
        num_samples=data_cfg.get("num_samples"),
    )

    model_cfg = cfg["model"]
    max_len = data_cfg["max_len"]
    beam_size = cfg.get("beam_size", 1)

    model = TransformerSeq2seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        max_len=max_len,
        dropout=model_cfg["dropout"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    val_dataset = val_loader.dataset
    n = min(num_samples, len(val_dataset))

    print(f"Evaluating first {n} val samples (beam_size={beam_size})\n")
    for i in range(n):
        src, tgt = val_dataset[i]
        src_text = ids_to_text(src.tolist(), src_vocab)
        ref_text = ids_to_text(tgt.tolist(), tgt_vocab)
        translation, _ = translate(
            src_text, model, src_vocab, tgt_vocab, device,
            max_len=max_len, beam_size=beam_size,
        )
        print(f"[{i+1}] SRC: {src_text}")
        print(f"     REF: {ref_text}")
        print(f"     HYP: {translation}\n")


def main():
    parser = argparse.ArgumentParser(description="WMT14 翻译推理")
    parser.add_argument("--text", type=str, help="待翻译文本（不传则进入交互模式）")
    parser.add_argument("--evaluate", action="store_true", help="在验证集上逐条展示翻译")
    parser.add_argument("--num_samples", type=int, default=20, help="evaluate 展示的条数")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--beam_size", type=int, default=None, help="覆盖 config 的 beam_size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    print(f"Using device: {device}")

    beam_size = args.beam_size if args.beam_size is not None else cfg.get("beam_size", 1)
    max_len = cfg["data"]["max_len"]

    model, src_vocab, tgt_vocab, cfg, device = load_vocab_and_model(args.checkpoint, args.config)
    print(f"src vocab: {len(src_vocab):,}  tgt vocab: {len(tgt_vocab):,}  beam_size={beam_size}\n")

    if args.evaluate:
        evaluate_dataset(args.checkpoint, args.config, num_samples=args.num_samples)
    elif args.text:
        translation, _ = translate(
            args.text, model, src_vocab, tgt_vocab, device,
            max_len=max_len, beam_size=beam_size,
        )
        print(f"SRC: {args.text}")
        print(f"OUT: {translation}")
    else:
        print(f"交互模式（beam_size={beam_size}，输入 'quit' 退出）\n")
        while True:
            try:
                text = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if text.strip().lower() in ("quit", "exit", "q"):
                break
            if not text.strip():
                continue
            translation, _ = translate(
                text, model, src_vocab, tgt_vocab, device,
                max_len=max_len, beam_size=beam_size,
            )
            print(f"  {translation}\n")


if __name__ == "__main__":
    main()
