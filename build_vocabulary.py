import json
import re
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

class LaTeXTokenizer:
    def __init__(self):
        self.pattern = re.compile(
            r'\\[a-zA-Z]+|'
            r'\\.|'
            r'[a-zA-Z]|'
            r'[0-9]|'
            r'\S'
        )
    
    def tokenize(self, latex_string: str) -> list:
        tokens = self.pattern.findall(latex_string)
        return tokens

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
    for token, freq in token_counter.most_common():
        if freq >= min_freq:
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

def load_vocabulary(vocab_file: str = 'tokens.json'):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    return vocab

if __name__ == "__main__":
    vocab, stats = build_vocabulary_from_dataset(
        cache_dir=r'D:\datasets',
        split='train',
        min_freq=1,
        output_file='tokens.json',
        max_latex_len=512
    )