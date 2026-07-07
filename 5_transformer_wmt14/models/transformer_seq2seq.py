import math
import torch
import torch.nn as nn


# ===========================
# 位置编码
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ===========================
# Transformer Seq2Seq（Encoder-Decoder）
# ===========================
class TransformerSeq2seq(nn.Module):
    """
    完整 Encoder-Decoder Transformer，用于机器翻译。
    """
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        max_len=256,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size

        # Embedding + 位置编码
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer（内置 encoder + decoder）
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # 输出投影
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def _generate_square_subsequent_mask(self, sz, device):
        # 因果 mask：上三角 -inf
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, src, tgt):
        """
        src: (B, src_len)  编码器输入
        tgt: (B, tgt_len)  解码器输入（teacher forcing，已 shift）
        """
        device = src.size(0)

        # Padding mask：True 表示 PAD 位置
        src_key_padding_mask = (src == 0)  # (B, src_len)
        tgt_key_padding_mask = (tgt == 0)  # (B, tgt_len)

        # 因果 mask
        tgt_len = tgt.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len, tgt.device)

        # Embedding + 位置编码
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Transformer 前向
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.output_proj(out)  # (B, tgt_len, tgt_vocab_size)
        return logits

    @torch.no_grad()
    def translate(self, src, src_vocab, tgt_vocab, max_len=64, beam_size=1):
        """
        推理：贪心 or Beam Search 生成目标序列
        src: (1, src_len) 单条源语句
        """
        self.eval()
        if beam_size == 1:
            return self._greedy_decode(src, tgt_vocab, max_len)
        else:
            return self._beam_search(src, tgt_vocab, max_len, beam_size)

    def _greedy_decode(self, src, tgt_vocab, max_len):
        device = src.device
        # Encoder 输出只算一次
        src_key_padding_mask = (src == 0)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        # 自回归生成
        ys = torch.tensor([[tgt_vocab.BOS_IDX]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1), device)
            tgt_emb = self.pos_encoder(self.tgt_embedding(ys) * math.sqrt(self.d_model))
            out = self.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = self.output_proj(out[:, -1, :])  # 只取最后一步
            next_word = torch.argmax(logits, dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            if next_word.item() == tgt_vocab.EOS_IDX:
                break
        return ys.squeeze(0).tolist()

    def _beam_search(self, src, tgt_vocab, max_len, beam_size):
        device = src.device
        src_key_padding_mask = (src == 0)
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        # 每条候选 = (序列 token list, 累计 log-prob)
        beams = [([tgt_vocab.BOS_IDX], 0.0)]
        completed = []

        for _ in range(max_len - 1):
            all_candidates = []
            for seq, score in beams:
                if seq[-1] == tgt_vocab.EOS_IDX:
                    completed.append((seq, score))
                    continue
                ys = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_mask = self._generate_square_subsequent_mask(len(seq), device)
                tgt_emb = self.pos_encoder(self.tgt_embedding(ys) * math.sqrt(self.d_model))
                out = self.transformer.decoder(
                    tgt=tgt_emb,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                logits = self.output_proj(out[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                topk_log_probs, topk_ids = log_probs.topk(beam_size)

                for i in range(beam_size):
                    new_seq = seq + [topk_ids[i].item()]
                    new_score = score + topk_log_probs[i].item()
                    all_candidates.append((new_seq, new_score))

            if not all_candidates:
                break

            # 按 score 排序，保留 top beam_size
            all_candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)  # 长度归一化
            beams = all_candidates[:beam_size]

        # 把未完成的也加入候选
        completed.extend(beams)
        if not completed:
            return beams[0][0]

        # 选长度归一化后最高分
        best = max(completed, key=lambda x: x[1] / len(x[0]))
        return best[0]
