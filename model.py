import torch
import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, hidden_dim=256, freeze_layers=True):
        super().__init__()
        
        resnet = models.resnet50()
        
        if freeze_layers:
            for param in resnet.layer1.parameters():
                param.requires_grad = False
            for param in resnet.layer2.parameters():
                param.requires_grad = False
            for param in resnet.layer3.parameters():
                param.requires_grad = False
            for param in resnet.layer4.parameters():
                param.requires_grad = True
        
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.projection = nn.Linear(2048, hidden_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x = self.projection(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=4, nhead=8, dropout=0.2, max_len=512):
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
        
        # Building mask
        self.register_buffer('causal_mask', torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())

    def forward(self, tgt, memory):
        B, T = tgt.shape
        tok_emb = self.token_emb(tgt)  # [B,T,H]
        pos = self.pos_emb(torch.arange(T, device=tgt.device))  # [T,H]
        pos = pos.unsqueeze(0).expand(B, -1, -1)
        tgt_emb = tok_emb + pos

        # Apply mask
        tgt_mask = self.causal_mask[:T, :T]

        out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)  # [B,T,H]
        return self.fc_out(out)

class FormulaRecognizer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, max_len=512):
        super().__init__()
        self.encoder = ResNetEncoder(hidden_dim, freeze_layers=True)
        self.decoder = TransformerDecoder(vocab_size, hidden_dim, max_len=max_len)
        self.max_len = max_len
        self.vocab_size = vocab_size

    def forward(self, images, tokens):
        memory = self.encoder(images)  # [B,S,H]
        out = self.decoder(tokens[:, :-1], memory)
        return out  # [B,T-1,V]

    def greedy_decode(self, image, vocab, device="cpu"):
        self.eval()
        with torch.no_grad():
            memory = self.encoder(image.unsqueeze(0).to(device))  # [1,S,H]

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
    
    def beam_search(self, image, vocab, device="cpu", beam_width=5, length_penalty=0.6):
        self.eval()
        with torch.no_grad():
            memory = self.encoder(image.unsqueeze(0).to(device))
            start_token = vocab.stoi[vocab.bos]
            end_token = vocab.stoi[vocab.eos]

            beams = [([start_token], 0.0)]
            completed = []

            step = 0
            continue_loop = True

            while continue_loop and step < self.max_len:
                candidates = []

                for tokens, score in beams:
                    if tokens[-1] == end_token:
                        completed.append((tokens, score))
                    else:
                        tokens_tensor = torch.tensor([tokens], device=device)
                        out = self.decoder(tokens_tensor, memory)
                        logits = out[:, -1, :]
                        log_probs = torch.log_softmax(logits, dim=-1)

                        top_log_probs, top_indices = log_probs.topk(beam_width)

                        for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                            new_tokens = tokens + [idx.item()]
                            new_score = score + log_prob.item()
                            candidates.append((new_tokens, new_score))

                if not candidates:
                    continue_loop = False
                else:
                    candidates.sort(
                        key=lambda x: x[1] / (len(x[0]) ** length_penalty),
                        reverse=True,
                    )
                    beams = candidates[:beam_width]
                    step += 1

                    if len(completed) >= beam_width:
                        continue_loop = False

            completed.extend(beams)
            completed.sort(
                key=lambda x: x[1] / (len(x[0]) ** length_penalty),
                reverse=True,
            )

            return completed[0][0] if completed else [start_token, end_token]   