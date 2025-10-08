from torch.utils.data import Dataset
import pandas as pd
import cv2
import ast
import os
import numpy as np
import torch

# Подгоняем изображения под общий размер, сохраняя отношение сторон. Паддим справа.
def resize_and_pad(img, target_height=64, target_width=256):
    # Проверяем, что изображение загружено корректно
    if img is None:
        raise ValueError("Не удалось загрузить изображение")
    
    h, w = img.shape
    if h == 0 or w == 0:
        raise ValueError("Изображение имеет нулевые размеры")
    
    scale = target_height / h
    new_w = int(w * scale)

    if new_w > target_width:
        new_w = target_width

    resized = cv2.resize(img, (new_w, target_height))

    padded = np.zeros((target_height, target_width), dtype=np.float32)
    padded[:, :new_w] = resized

    return padded


# ===== Dataset =====
class FormulaDataset(Dataset):
    def __init__(self, path, vocab, transform=None):
        self.data = pd.read_csv(os.path.join(path, 'annotations.csv'))
        self.img_dir = os.path.join(path, 'images')
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # --- картинка ---
        img_path = os.path.join(self.img_dir, row['filenames'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Проверяем, что изображение загружено
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        
        img = resize_and_pad(img)
        img = img / 255.0 # нормируем
        img = torch.tensor(img).unsqueeze(0).float()  # (1,H,W)

        # --- токены ---
        try:
            info = ast.literal_eval(row['image_data'])
            tokens = info['full_latex_chars']   # список строк-токенов
            token_ids = self.vocab.encode(tokens)
        except (ValueError, SyntaxError, KeyError) as e:
            raise ValueError(f"Ошибка при обработке токенов для строки {idx}: {e}")

        return img, torch.tensor(token_ids)

# ===== Vocabulary =====
class Vocab:
    def __init__(self, token_list):

        # спец символы
        self.pad = "<pad>"
        self.bos = "<bos>"
        self.eos = "<eos>"
        self.unk = "<unk>"

        self.tokens = [self.pad, self.bos, self.eos, self.unk] + sorted(token_list)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}

        self.length = len(self.tokens)  # ATTENTION!!!

    def __len__(self):
        return self.length

    def encode(self, token_seq):
        if not isinstance(token_seq, list):
            raise ValueError(f"Expected list of tokens, got {type(token_seq)}")
        
        if not all(isinstance(t, str) for t in token_seq):
            raise ValueError("All tokens must be strings")
        
        return [self.stoi[self.bos]] + \
               [self.stoi.get(t, self.stoi[self.unk]) for t in token_seq] + \
               [self.stoi[self.eos]]

    def decode(self, ids):
        # Декодируем и обрезаем по первому <eos>.
        eos_id = self.stoi[self.eos]
        pad_id = self.stoi[self.pad]
        bos_id = self.stoi[self.bos]

        toks = []
        for i in ids:
            if i == eos_id:
                break
            # Пропускаем pad и bos токены
            if i not in (pad_id, bos_id):
                toks.append(self.itos[i])

        return " ".join(toks)
