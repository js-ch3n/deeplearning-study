import time
import logging
from datetime import timedelta
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# ── 日志 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================
# 词汇表
# ===========================
class Vocab:
    def __init__(self, max_size=37000):
        self.max_size = max_size
        # 特殊 token
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}

    def build(self, texts):
        from collections import Counter
        counter = Counter()
        for text in tqdm(texts, desc="构建词汇表", unit="句"):
            counter.update(text.split())

        most_common = counter.most_common(self.max_size - 4)
        for idx, (word, _) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=64):
        tokens = text.split()
        indices = [self.BOS_IDX]
        indices += [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        indices.append(self.EOS_IDX)
        if len(indices) < max_len:
            indices += [self.PAD_IDX] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            indices[-1] = self.EOS_IDX
        return indices

    def __len__(self):
        return len(self.word2idx)


# ===========================
# 数据集
# ===========================
class WMT14Dataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_len=64):
        """
        data: list of dict, 每个 dict 形如 {"translation": {"en": "...", "de": "..."}}
        """
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        src_text = self.data[index]["translation"]["en"]
        tgt_text = self.data[index]["translation"]["de"]
        src_encoded = self.src_vocab.encode(src_text, self.max_len)
        tgt_encoded = self.tgt_vocab.encode(tgt_text, self.max_len)
        return (
            torch.tensor(src_encoded, dtype=torch.long),
            torch.tensor(tgt_encoded, dtype=torch.long),
        )


# ===========================
# 加载数据
# ===========================
def load_wmt14_data(split="train", lang_pair="de-en", num_samples=None, repo_id="wmt/wmt14",
                    cache_dir="../data/hf_cache"):
    """
    通过 HuggingFace datasets 加载 WMT14。
    split: "train" / "validation" / "test"
    lang_pair: "de-en" (英<->德) 或 "fr-en" (英<->法)
    num_samples: 只取前 N 条，用于调试；None 表示全量
    cache_dir: 本地缓存目录，已下载过则直接读取不重下
    """
    t0 = time.perf_counter()
    log.info(f"[{split}] 开始加载 WMT14 {lang_pair} (cache: {cache_dir}) ...")

    with tqdm(total=None, desc=f"[{split}] 加载数据集", unit="样本") as pbar:
        dataset = load_dataset(repo_id, lang_pair, split=split, cache_dir=cache_dir)
        pbar.update(len(dataset))  # 加载完成后更新总量

    elapsed = time.perf_counter() - t0
    log.info(f"[{split}] 加载完成，共 {len(dataset):,} 条，耗时 {timedelta(seconds=int(elapsed))}")

    if num_samples is not None:
        n = min(num_samples, len(dataset))
        dataset = dataset.select(range(n))
        log.info(f"[{split}] 截取前 {n} 条用于调试 （原 {len(dataset)+num_samples-n:,} 条）")

    return dataset


def get_loaders(
    lang_pair="de-en",
    batch_size=64,
    max_len=64,
    vocab_size=37000,
    num_samples=None,
    num_workers=16,
):
    """
    返回 train_loader, val_loader, src_vocab, tgt_vocab
    num_samples: 调试时只取前 N 条；None 表示全量
    """
    total_t0 = time.perf_counter()

    log.info("=" * 50)
    log.info(f" lang_pair={lang_pair}, batch_size={batch_size}, max_len={max_len}, vocab_size={vocab_size}, num_samples={num_samples}")
    log.info("=" * 50)

    train_data = load_wmt14_data("train", lang_pair, num_samples)
    val_data = load_wmt14_data("validation", lang_pair, num_samples)

    # 词汇表
    log.info("[vocab] 构建源语言(en) 词汇表 ...")
    src_vocab = Vocab(max_size=vocab_size)
    src_vocab.build([item["translation"]["en"] for item in train_data])
    log.info(f"[vocab] 源语言词汇表大小: {len(src_vocab):,}")

    log.info("[vocab] 构建目标语言(de) 词汇表 ...")
    tgt_vocab = Vocab(max_size=vocab_size)
    tgt_vocab.build([item["translation"]["de"] for item in train_data])
    log.info(f"[vocab] 目标语言词汇表大小: {len(tgt_vocab):,}")

    train_dataset = WMT14Dataset(train_data, src_vocab, tgt_vocab, max_len)
    val_dataset = WMT14Dataset(val_data, src_vocab, tgt_vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total_elapsed = time.perf_counter() - total_t0
    log.info("=" * 50)
    log.info(f" 全部完成！总耗时 {timedelta(seconds=int(total_elapsed))}")
    log.info(f" 训练集 {len(train_dataset):,} 条, 验证集 {len(val_dataset):,} 条")
    log.info("=" * 50)

    return train_loader, val_loader, src_vocab, tgt_vocab


# ===========================
# 测试
# ===========================
if __name__ == "__main__":
    # 先用 5000 条快速验证
    train_loader, val_loader, src_vocab, tgt_vocab = get_loaders(num_samples=10000)
    for src, tgt in train_loader:
        print("src shape:", src.shape, "tgt shape:", tgt.shape)
        break
    print("源语言词汇表大小：", len(src_vocab))
    print("目标语言词汇表大小：", len(tgt_vocab))
    print("训练集数量：", len(train_loader.dataset))
    print("验证集数量：", len(val_loader.dataset))
