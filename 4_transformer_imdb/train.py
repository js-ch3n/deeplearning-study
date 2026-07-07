import os
import json
import datetime

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.imdb_dataset import get_loaders
from models.transformer_model import TransformerClassifier

import matplotlib.pyplot as plt


# ===========================
# 加载配置
# ===========================
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================
# 设备
# ===========================
def resolve_device(device_str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ===========================
# 数据加载
# ===========================
def load_data(data_folder, batch_size=64, max_len=256, vocab_size=25000, num_samples=None):
    train_loader, test_loader, vocab = get_loaders(
        data_dir=data_folder,
        batch_size=batch_size,
        max_len=max_len,
        vocab_size=vocab_size,
        num_samples=num_samples,
    )
    return train_loader, test_loader, len(vocab)


# ===========================
# 绘图
# ===========================
def plot_losses(train_loss_list, test_loss_list, save_path=None):
    plt.figure(figsize=(8, 5))

    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("IMDB Training Curve")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")

    plt.show()


# ===========================
# 测试（使用已加载的 model 和 test_loader）
# ===========================
def test(model, test_loader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            loss = criterion(output, y)
            total_loss += loss.item()

            pred = torch.argmax(output, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    return avg_loss, accuracy


# ===========================
# 测试（独立模式，加载模型文件）
# ===========================
def test_from_file(model_path, config_path="config.yaml"):
    cfg = load_config(config_path)
    device = resolve_device(cfg["device"])

    data_cfg = cfg.get("data", {})
    if isinstance(data_cfg, dict):
        data_folder = data_cfg.get("folder", "./data/aclImdb")
    else:
        data_folder = str(data_cfg)

    model_cfg = cfg.get("model", {})
    max_len = model_cfg.get("max_len", 256)
    vocab_size = model_cfg.get("vocab_size", 25000)

    _, test_loader, vocab_size = load_data(
        data_folder, batch_size=cfg["batch_size"],
        max_len=max_len, vocab_size=vocab_size,
    )

    model = TransformerClassifier(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    avg_loss, accuracy = test(model, test_loader, device)
    print(f"Test Loss: {avg_loss:.4f}  Test Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy


# ===========================
# Checkpoint 保存辅助
# ===========================
def make_checkpoint_dict(epoch, model, optimizer, scheduler,
                         train_loss, test_loss, test_acc):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def save_checkpoint(checkpoint_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint_dict, path)


# ===========================
# 训练单轮
# ===========================
def train_one_epoch(model, train_loader, test_loader,
                    criterion, optimizer, scheduler, device):
    model.train()

    total_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    scheduler.step()

    test_loss, test_acc = test(model, test_loader, device)

    return train_loss, test_loss, test_acc


# ===========================
# 主训练流程
# ===========================
def train():
    # ----- 加载配置 -----
    cfg = load_config("config.yaml")

    seed = cfg["seed"]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    learning_rate = cfg["learning_rate"]

    data_cfg = cfg.get("data", {})
    if isinstance(data_cfg, dict):
        data_folder = data_cfg.get("folder", "./data/aclImdb")
    else:
        data_folder = str(data_cfg)

    model_cfg = cfg.get("model", {})
    max_len = model_cfg.get("max_len", 256)
    vocab_size = model_cfg.get("vocab_size", 25000)
    d_model = model_cfg.get("d_model", 128)
    nhead = model_cfg.get("nhead", 8)
    num_layers = model_cfg.get("num_layers", 3)
    dim_feedforward = model_cfg.get("dim_feedforward", 512)
    dropout = model_cfg.get("dropout", 0.1)
    num_classes = model_cfg.get("num_classes", 2)

    ckpt_cfg = cfg["checkpoint"]
    save_every_n = ckpt_cfg["save_every_n_epochs"]
    save_best = ckpt_cfg["save_best"]
    save_last = ckpt_cfg["save_last"]
    root_dir = ckpt_cfg["root_dir"]
    archive_under_runs = ckpt_cfg["archive_under_runs"]

    runs_dir = cfg["runs_dir"]
    log_metrics = cfg["log_metrics"]
    save_summary_flag = cfg["save_summary"]

    # ----- 随机种子 -----
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ----- 设备 -----
    device = resolve_device(cfg["device"])
    print(f"Using device: {device}")

    # ----- 运行目录 -----
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"run_{run_timestamp}")
    run_ckpt_dir = os.path.join(run_dir, "checkpoints")
    run_plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(run_ckpt_dir, exist_ok=True)
    os.makedirs(run_plots_dir, exist_ok=True)
    os.makedirs(root_dir, exist_ok=True)

    # 保存运行时配置快照
    run_cfg = dict(cfg)
    run_cfg["_start_time"] = datetime.datetime.now().isoformat(timespec="seconds")
    run_cfg["_device_resolved"] = str(device)
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        run_cfg["_git_commit"] = git_commit
    except Exception:
        run_cfg["_git_commit"] = ""
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(run_cfg, f, default_flow_style=False, allow_unicode=True)

    # ----- 模型/优化器/调度器 -----
    train_loader, test_loader, vocab_size = load_data(
        data_folder, batch_size=batch_size,
        max_len=max_len, vocab_size=vocab_size,
    )

    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_name = cfg["optimizer"]["name"]
    weight_decay = cfg["optimizer"].get("weight_decay", 0.0)
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        momentum = cfg["optimizer"].get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler_name = cfg["scheduler"]["name"]
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["scheduler"]["step_size"],
            gamma=cfg["scheduler"]["gamma"]
        )
    elif scheduler_name == "CosineAnnealingLR":
        t_max = cfg["scheduler"].get("T_max", epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # ----- 训练循环 -----
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []

    best_acc = 0.0
    best_epoch = 0
    metrics_file = None
    if log_metrics:
        metrics_file = open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8")

    for epoch in range(epochs):
        train_loss, test_loss, test_acc = train_one_epoch(
            model, train_loader, test_loader,
            criterion, optimizer, scheduler, device
        )

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Test Accuracy: {test_acc:.4f} "
            f"LR: {current_lr:.6f}"
        )

        # 指标日志
        if metrics_file:
            metrics_line = json.dumps({
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "test_loss": round(test_loss, 6),
                "test_acc": round(test_acc, 6),
                "lr": current_lr,
            })
            metrics_file.write(metrics_line + "\n")
            metrics_file.flush()

        # 构建 checkpoint dict
        ckpt_dict = make_checkpoint_dict(
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            test_loss=test_loss,
            test_acc=test_acc,
        )

        # 每 N 轮保存
        if (epoch + 1) % save_every_n == 0:
            save_checkpoint(ckpt_dict, os.path.join(root_dir, f"epoch_{epoch + 1}.pth"))
            if archive_under_runs:
                save_checkpoint(ckpt_dict, os.path.join(run_ckpt_dir, f"epoch_{epoch + 1}.pth"))

        # 最佳模型
        if save_best and test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            save_checkpoint(ckpt_dict, os.path.join(root_dir, "best.pth"))
            if archive_under_runs:
                save_checkpoint(ckpt_dict, os.path.join(run_ckpt_dir, "best.pth"))

        # 最后一轮（每轮覆盖）
        if save_last:
            save_checkpoint(ckpt_dict, os.path.join(root_dir, "last.pth"))
            if archive_under_runs:
                save_checkpoint(ckpt_dict, os.path.join(run_ckpt_dir, "last.pth"))

    if metrics_file:
        metrics_file.close()

    # ----- 保留旧行为：保存最终模型 -----
    torch.save(model.state_dict(), os.path.join(root_dir, "transformer_imdb.pth"))
    print("训练完成，模型已保存。")

    # ----- 保存汇总 -----
    if save_summary_flag:
        summary = {
            "best_acc": round(best_acc, 6),
            "best_epoch": best_epoch,
            "final_train_loss": round(train_loss_list[-1], 6),
            "final_test_loss": round(test_loss_list[-1], 6),
            "final_test_acc": round(test_acc_list[-1], 6),
            "end_time": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # ----- 绘图 -----
    plot_losses(
        train_loss_list,
        test_loss_list,
        save_path=os.path.join(run_plots_dir, "losses.png"),
    )


if __name__ == "__main__":
    train()
