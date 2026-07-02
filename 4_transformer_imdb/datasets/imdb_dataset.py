import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
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
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        most_common = counter.most_common(self.max_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=256):
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(t, 1) for t in tokens]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices

    def __len__(self):
        return len(self.word2idx)

    @staticmethod
    def tokenize(text):
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
def load_imdb_data(data_dir="./data/aclImdb"):
    import os

    def read_folder(folder, label):
        texts = []
        labels = []
        for fname in os.listdir(folder):
            if fname.endswith(".txt"):
                with open(os.path.join(folder, fname), encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(label)
        return texts, labels

    train_pos = os.path.join(data_dir, "train", "pos")
    train_neg = os.path.join(data_dir, "train", "neg")
    test_pos = os.path.join(data_dir, "test", "pos")
    test_neg = os.path.join(data_dir, "test", "neg")

    train_texts, train_labels = read_folder(train_pos, 1)
    neg_texts, neg_labels = read_folder(train_neg, 0)
    train_texts += neg_texts
    train_labels += neg_labels

    test_texts, test_labels = read_folder(test_pos, 1)
    neg_texts, neg_labels = read_folder(test_neg, 0)
    test_texts += neg_texts
    test_labels += neg_labels

    return train_texts, train_labels, test_texts, test_labels


def get_loaders(data_dir="./data/aclImdb", batch_size=64, max_len=256, vocab_size=25000):
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(data_dir)

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
    train_loader, test_loader, vocab = get_loaders()
    for x, y in train_loader:
        print(x.shape, y.shape)
        break
    print("词汇表大小：", len(vocab))
    print("训练集数量：", len(train_loader.dataset))
    print("测试集数量：", len(test_loader.dataset))
