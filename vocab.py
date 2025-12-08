import json
import os
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from latex_tokenizer import LaTeXTokenizer

def build_vocabulary_from_dataset(
    cache_dir: str = r'D:\datasets',
    split: str = 'train',
    min_freq: int = 1,
    output_file: str = 'tokens.json',
    max_latex_len: int = 512
):
    print(f"Loading dataset...")
    ds = load_dataset(
        "hoang-quoc-trung/fusion-image-to-latex-datasets",
        cache_dir=cache_dir,
        split=split
    )

    print(f"Filtering by LaTeX length <= {max_latex_len}...")
    ds = ds.filter(lambda x: len(x['latex']) <= max_latex_len)

    tokenizer = LaTeXTokenizer()
    token_counter = Counter()

    print(f"Tokenizing {len(ds)} samples...")
    for item in tqdm(ds):
        latex = item['latex']
        tokens = tokenizer.tokenize(latex)
        token_counter.update(tokens)

    print(f"Found {len(token_counter)} unique tokens")

    vocab = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3
    }

    idx = 4
    for token in tokenizer.common_tokens:
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    for token, freq in token_counter.most_common():
        if freq >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1

    print(f"Final vocabulary size: {len(vocab)} tokens")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary saved to {output_file}")

    stats = {
        'total_tokens': len(vocab),
        'min_frequency': min_freq,
        'most_common_tokens': token_counter.most_common(30)
    }

    return vocab, stats

class Vocab:
    def __init__(self, vocab_path=None, token_list=None, **kwargs):
        self.pad = "<PAD>"
        self.bos = "<SOS>"
        self.eos = "<EOS>"
        self.unk = "<UNK>"
        
        if vocab_path and os.path.exists(vocab_path):
            self._load_from_file(vocab_path)
        elif token_list:
            self._build_from_tokens(token_list)
        else:
            print(f"Vocab file not found at '{vocab_path}'. Building from dataset...")
            save_path = vocab_path if vocab_path else 'tokens.json'
            
            vocab_dict, _ = build_vocabulary_from_dataset(output_file=save_path, **kwargs)
            
            self.stoi = vocab_dict
            self.itos = {i: t for t, i in self.stoi.items()}
            self.tokens = [self.itos[i] for i in range(len(self.itos))]
        
        self.length = len(self.tokens)

    def _load_from_file(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            loaded_vocab = json.load(f)
        
        if self.pad in loaded_vocab and self.bos in loaded_vocab and \
           self.eos in loaded_vocab and self.unk in loaded_vocab:
            self.stoi = loaded_vocab
            self.itos = {i: t for t, i in self.stoi.items()}
            self.tokens = [self.itos[i] for i in range(len(self.itos))]
        else:
            token_list = list(loaded_vocab.keys())
            self._build_from_tokens(token_list)

    def _build_from_tokens(self, token_list):
        special_tokens = [self.pad, self.bos, self.eos, self.unk]
        filtered_tokens = [t for t in token_list if t not in special_tokens]
        self.tokens = special_tokens + sorted(set(filtered_tokens))
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}

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
        eos_id = self.stoi[self.eos]
        pad_id = self.stoi[self.pad]
        bos_id = self.stoi[self.bos]
        
        toks = []
        for i in ids:
            if i == eos_id:
                break
            if i not in (pad_id, bos_id):
                toks.append(self.itos[i])
        
        return "".join(toks)

    def save(self, vocab_path):
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)
