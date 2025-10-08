import torch.nn as nn
import torch

# --- CNN encoder ---
class CNNEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(),
        )

    def forward(self, x):
        feats = self.conv(x)  # [B,Hid,h,w]
        B, C, H, W = feats.shape
        # Преобразуем в последовательность: [B, H*W, C]
        feats = feats.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return feats  # [B, seq_len, hidden_dim]


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4, nhead=8, dropout=0.1, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.max_len = max_len

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Создаем маску один раз и кэшируем
        self.register_buffer('causal_mask', torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())

    def forward(self, tgt, memory):
        B, T = tgt.shape
        tok_emb = self.token_emb(tgt)  # [B,T,H]
        pos = self.pos_emb(torch.arange(T, device=tgt.device))  # [T,H]
        pos = pos.unsqueeze(0).expand(B, -1, -1)
        tgt_emb = tok_emb + pos

        # Используем маску
        tgt_mask = self.causal_mask[:T, :T]

        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)  # [B,T,H]
        return self.fc_out(out)
    

# Итоговая модель 
class FormulaRecognizer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, max_len=512):
        super().__init__()
        self.encoder = CNNEncoder(hidden_dim)
        self.decoder = TransformerDecoder(vocab_size, hidden_dim, max_len=max_len)
        self.max_len = max_len
        self.vocab_size = vocab_size

    def forward(self, images, tokens):
        memory = self.encoder(images)  # [B,S,H]
        out = self.decoder(tokens[:, :-1], memory)  # предсказываем без последнего токена
        return out  # [B,T-1,V]

    def greedy_decode(self, image, vocab, device="cpu"):
        self.eval()
        with torch.no_grad():
            memory = self.encoder(image.unsqueeze(0).to(device))  # [1,S,H]

            # Используем правильные токены из словаря
            start_token = vocab.stoi[vocab.bos]
            end_token = vocab.stoi[vocab.eos]
            
            tokens = torch.tensor([[start_token]], device=device)
            for _ in range(self.max_len):
                out = self.decoder(tokens, memory)  # [1,T,V]
                next_token = out[:, -1, :].argmax(-1)  # [1]
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == end_token:
                    break

        return tokens.squeeze(0).tolist()