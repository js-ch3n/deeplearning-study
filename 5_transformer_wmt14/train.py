import os
import json
import datetime

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from data.wmt14_dataset import get_loaders
from models.transformer_seq2seq import TransformerSeq2seq


# ===========================
# 配置 & 设备
# ===========================
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ===========================
# 数据
# ===========================
def load_data(lang_pair, batch_size, max_len, vocab_size, num_samples):
    train_loader, val_loader, src_vocab, tgt_vocab = get_loaders(
        lang_pair=lang_pair,
        batch_size=batch_size,
        max_len=max_len,
        vocab_size=vocab_size,
        num_samples=num_samples,
    )
    return train_loader, val_loader, src_vocab, tgt_vocab


# ===========================
# 指标：困惑度
# ===========================
def compute_ppl(loss):
    return torch.exp(torch.tensor(loss)).item()


# ===========================
# 绘图
# ===========================
def plot_losses(train_loss_list, val_loss_list, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross-Entropy)")
    plt.title("WMT14 Training Curve")
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


# ===========================
# 验证
# ===========================
def validate(model, val_loader, criterion, device, tgt_pad_idx=0):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # teacher forcing：tgt_input 去掉最后一位，tgt_out 去掉第一位
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src, tgt_input)  # (B, tgt_len-1, vocab_size)
            B, L, V = logits.shape
            logits_flat = logits.reshape(-1, V)
            tgt_flat = tgt_out.reshape(-1)

            loss = criterion(logits_flat, tgt_flat)
            # 只统计非 PAD token 的 loss
            non_pad = (tgt_flat != tgt_pad_idx).sum().item()
            total_loss += loss.item() * max(non_pad, 1)
            total_tokens += max(non_pad, 1)

            # token 准确率（忽略 PAD）
            preds = torch.argmax(logits_flat, dim=1)
            mask = (tgt_flat != tgt_pad_idx)
            correct += ((preds == tgt_flat) & mask).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    token_acc = correct / max(total, 1)
    return avg_loss, token_acc


# ===========================
# 单轮训练
# ===========================
def train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                    device, tgt_pad_idx=0, epoch=None, total_epochs=None,
                    plateau_scheduler=False):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    total = 0

    desc = f"Epoch {epoch + 1}/{total_epochs}" if epoch is not None and total_epochs is not None else "Train"
    pbar = tqdm(train_loader, desc=desc, unit="batch", leave=True)

    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_input)
        B, L, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        tgt_flat = tgt_out.reshape(-1)

        loss = criterion(logits_flat, tgt_flat)
        non_pad = (tgt_flat != tgt_pad_idx).sum().item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * max(non_pad, 1)
        total_tokens += max(non_pad, 1)

        preds = torch.argmax(logits_flat, dim=1)
        mask = (tgt_flat != tgt_pad_idx)
        correct += ((preds == tgt_flat) & mask).sum().item()
        total += mask.sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = total_loss / max(total_tokens, 1)
    train_acc = correct / max(total, 1)

    if plateau_scheduler:
        # plateau 用 val loss，在外层 step
        pass
    else:
        scheduler.step()

    return train_loss, train_acc


# ===========================
# Checkpoint 辅助
# ===========================
def make_checkpoint_dict(epoch, model, optimizer, scheduler, train_loss, val_loss):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }


def save_checkpoint(checkpoint_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint_dict, path)


# ===========================
# 主训练
# ===========================
def train():
    cfg = load_config("config.yaml")

    seed = cfg["seed"]
    batch_size = cfg["data"]["batch_size"]
    epochs = cfg["epochs"]
    learning_rate = cfg["learning_rate"]
    lang_pair = cfg["data"]["lang_pair"]
    max_len = cfg["data"]["max_len"]
    vocab_size = cfg["data"]["vocab_size"]
    num_samples = cfg["data"].get("num_samples")

    model_cfg = cfg["model"]
    ckpt_cfg = cfg["checkpoint"]
    runs_dir = cfg["runs_dir"]
    log_metrics = cfg["log_metrics"]
    save_summary_flag = cfg["save_summary"]
    scheduler_name = cfg["scheduler"]["name"]
    use_plateau = scheduler_name == "ReduceLROnPlateau"

    # 随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(cfg["device"])
    print(f"Using device: {device}")

    # 运行目录
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"run_{run_ts}")
    run_ckpt_dir = os.path.join(run_dir, "checkpoints")
    run_plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(run_plots_dir, exist_ok=True)
    os.makedirs(ckpt_cfg["root_dir"], exist_ok=True)

    # 配置快照
    run_cfg = dict(cfg)
    run_cfg["_start_time"] = datetime.datetime.now().isoformat(timespec="seconds")
    run_cfg["_device_resolved"] = str(device)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(run_cfg, f, default_flow_style=False, allow_unicode=True)

    # 数据
    train_loader, val_loader, src_vocab, tgt_vocab = load_data(
        lang_pair, batch_size, max_len, vocab_size, num_samples
    )
    print(f"src vocab: {len(src_vocab):,}  tgt vocab: {len(tgt_vocab):,}")

    # 模型
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # 损失：忽略 PAD
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 优化器
    optimizer_name = cfg["optimizer"]["name"]
    weight_decay = cfg["optimizer"].get("weight_decay", 0.0)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = cfg["optimizer"].get("momentum", 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # 调度器
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["scheduler"]["step_size"], gamma=cfg["scheduler"]["gamma"])
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler"].get("T_max", epochs))
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg["scheduler"].get("factor", 0.5),
            patience=cfg["scheduler"].get("patience", 3),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # 训练循环
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float("inf")
    best_epoch = 0
    metrics_file = None
    if log_metrics:
        metrics_file = open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            tgt_pad_idx=0,
            epoch=epoch, total_epochs=epochs,
            plateau_scheduler=use_plateau,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  "
            f"Val PPL: {compute_ppl(val_loss):.2f}  "
            f"LR: {current_lr:.6f}"
        )

        if use_plateau:
            scheduler.step(val_loss)

        if metrics_file:
            metrics_file.write(json.dumps({
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_ppl": round(compute_ppl(val_loss), 4),
                "lr": current_lr,
            }) + "\n")
            metrics_file.flush()

        ckpt_dict = make_checkpoint_dict(
            epoch=epoch + 1, model=model, optimizer=optimizer,
            scheduler=scheduler, train_loss=train_loss, val_loss=val_loss,
        )
        save_every = ckpt_cfg["save_every_n_epochs"]
        root_dir = ckpt_cfg["root_dir"]

        if (epoch + 1) % save_every == 0:
            save_checkpoint(ckpt_dict, os.path.join(root_dir, f"epoch_{epoch + 1}.pth"))
            if ckpt_cfg["archive_under_runs"]:
                save_checkpoint(ckpt_dict, os.path.join(run_ckpt_dir, f"epoch_{epoch + 1}.pth"))

        if ckpt_cfg["save_best"] and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            save_checkpoint(ckpt_dict, os.path.join(root_dir, "best.pth"))
            if ckpt_cfg["archive_under_runs"]:
                save_checkpoint(ckpt_dict, os.path.join(run_ckpt_dir, "best.pth"))

        if ckpt_cfg["save_last"]:
            save_checkpoint(ckpt_dict, os.path.join(root_dir, "last.pth"))
            if ckpt_cfg["archive_under_runs"]:
                save_checkpoint(ckpt_dict, os.path.join(run_ckpt_dir, "last.pth"))

    if metrics_file:
        metrics_file.close()

    torch.save(model.state_dict(), os.path.join(root_dir, "transformer_wmt14.pth"))
    print(f"训练完成，最佳 epoch {best_epoch}，val loss {best_val_loss:.4f}")

    if save_summary_flag:
        summary = {
            "best_val_loss": round(best_val_loss, 6),
            "best_epoch": best_epoch,
            "final_train_loss": round(train_loss_list[-1], 6),
            "final_val_loss": round(val_loss_list[-1], 6),
            "final_val_ppl": round(compute_ppl(val_loss_list[-1]), 4),
            "end_time": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_losses(
        train_loss_list, val_loss_list,
        save_path=os.path.join(run_plots_dir, "losses.png"),
    )


if __name__ == "__main__":
    train()
