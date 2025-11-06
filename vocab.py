import json

class Vocab:
    def __init__(self, vocab_path=None, token_list=None):
        self.pad = "<PAD>"
        self.bos = "<SOS>"
        self.eos = "<EOS>"
        self.unk = "<UNK>"
        
        if vocab_path:
            self._load_from_file(vocab_path)
        elif token_list:
            self._build_from_tokens(token_list)
        else:
            raise ValueError("Either vocab_path or token_list must be provided")
        
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
