import torch
import torch.nn as nn
import math


# ===========================
# 位置编码
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ===========================
# Transformer编码器层（pre-norm，继承 nn.TransformerEncoderLayer）
# ===========================
class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
            bias=True,
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        # pre-norm + 残差（覆盖默认 post-norm）
        src2 = self.self_attn(src, src, src,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


# ===========================
# Transformer分类模型
# ===========================
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, max_len=256, dropout=0.1, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, seq_len)
        mask = (x == 0)  # padding mask

        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # 取 <CLS> 位置 (这里用第一个token) 或 mean pooling
        x = x.mean(dim=1)  # (B, d_model)

        x = self.classifier(x)
        return x
