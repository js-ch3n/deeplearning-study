import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm.auto import tqdm
import re


# ===========================
# 词汇表
# ===========================
class Vocab:
    def __init__(self, max_size=25000):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        counter = Counter()
        for text in tqdm(texts, desc="构建词汇表", unit="doc"):
            tokens = self.tokenize(text)
            counter.update(tokens)

        most_common = counter.most_common(self.max_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=256):
        # 编码时，默认最大长度为256
        # 分词（小写、去标点符号）
        tokens = self.tokenize(text)

        indices = [self.word2idx.get(t, 1) for t in tokens]
        # 不够最大，补PAD
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        # 超最大token数，截断
        else:
            indices = indices[:max_len]
        return indices

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text):
        # 去小写
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        return text.split()


# ===========================
# 数据集
# ===========================
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.vocab.encode(self.texts[index], self.max_len)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


# ===========================
# 加载数据
# ===========================
def load_imdb_data(data_dir="./data/aclImdb", num_samples=None):
    import os

    def read_folder(folder, label, limit=None):
        texts = []
        labels = []
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
        if limit:
            files = files[:limit]
        for fname in tqdm(files, desc=f"读取 {os.path.basename(folder)}", unit="file"):
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
        return texts, labels

    # num_samples 控制每类读取数量，None 表示全量
    per_class = num_samples // 2 if num_samples else None

    train_pos = os.path.join(data_dir, "train", "pos")
    train_neg = os.path.join(data_dir, "train", "neg")
    test_pos = os.path.join(data_dir, "test", "pos")
    test_neg = os.path.join(data_dir, "test", "neg")

    train_texts, train_labels = read_folder(train_pos, 1, per_class)
    neg_texts, neg_labels = read_folder(train_neg, 0, per_class)
    train_texts += neg_texts
    train_labels += neg_labels

    test_texts, test_labels = read_folder(test_pos, 1, per_class)
    neg_texts, neg_labels = read_folder(test_neg, 0, per_class)
    test_texts += neg_texts
    test_labels += neg_labels

    return train_texts, train_labels, test_texts, test_labels


def get_loaders(data_dir="./data/aclImdb", batch_size=64, max_len=256, vocab_size=25000, num_samples=None):
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(data_dir, num_samples=num_samples)

    vocab = Vocab(max_size=vocab_size)
    vocab.build(train_texts)

    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab


# ===========================
# 测试
# ===========================
if __name__ == "__main__":
    train_loader, test_loader, vocab = get_loaders(num_samples=10)
    for x, y in train_loader:
        print(x.shape, y.shape)
        break
    print("词汇表大小：", len(vocab))
    print("训练集数量：", len(train_loader.dataset))
    print("测试集数量：", len(test_loader.dataset))
