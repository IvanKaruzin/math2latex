import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from datasets import load_dataset
from torchvision import transforms
from typing import Optional
from preprosess_image import ImagePreprocessing
from latex_tokenizer import LaTeXTokenizer
from config import IMAGE_SIZE, MAX_LATEX_LEN, HF_CACHE_DIR, IMAGES_DIR

class LaTeXDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        vocab,
        cache_dir: str = HF_CACHE_DIR,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        max_latex_len: int = MAX_LATEX_LEN
    ):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.tokenizer = LaTeXTokenizer()
        self.transform = transform or self._default_transform()
        
        print(f"Loading dataset from HuggingFace...")
        ds = load_dataset(
            "hoang-quoc-trung/fusion-image-to-latex-datasets",
            cache_dir=cache_dir,
            split=split
        )
        
        print(f"Filtering by LaTeX length <= {max_latex_len}...")
        ds = ds.filter(lambda x: len(x['latex']) <= max_latex_len)
        
        self.filenames = ds['image_filename']
        self.latex_formulas = ds['latex']
        
        print(f"Dataset ready: {len(self.filenames)} samples")

    def _default_transform(self):
        return ImagePreprocessing(
            target_size=IMAGE_SIZE, 
            fill=255,
            to_tensor=True,
            normalize=True
        )


    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image_filename = self.filenames[idx]
        latex = self.latex_formulas[idx]
        img_path = self.images_dir / image_filename
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            image = torch.zeros((3, 224, 224))
        
        tokens = self.tokenizer.tokenize(latex)
        encoded_tokens = self.vocab.encode(tokens)
        token_tensor = torch.LongTensor(encoded_tokens)
        
        return image, token_tensor

def collate_fn(batch):
    images, token_tensors = zip(*batch)
    
    images = torch.stack(images, dim=0)
    
    max_len = max(len(t) for t in token_tensors)
    pad_id = 0
    
    padded_tokens = []
    for tokens in token_tensors:
        padding = torch.full((max_len - len(tokens),), pad_id, dtype=torch.long)
        padded = torch.cat([tokens, padding])
        padded_tokens.append(padded)
    
    token_batch = torch.stack(padded_tokens, dim=0)
    
    return images, token_batch

def get_dataloaders(
    images_dir: str = IMAGES_DIR,
    vocab=None,
    cache_dir: str = HF_CACHE_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
    max_latex_len: int = 512
):

    train_dataset = LaTeXDataset(
        images_dir=images_dir,
        vocab=vocab,
        cache_dir=cache_dir,
        split='train[:5%]',
        max_latex_len=max_latex_len
    )
    
    val_dataset = LaTeXDataset(
        images_dir=images_dir,
        vocab=vocab,
        cache_dir=cache_dir,
        split='validation',
        max_latex_len=max_latex_len
    )
    
    test_dataset = LaTeXDataset(
        images_dir=images_dir,
        vocab=vocab,
        cache_dir=cache_dir,
        split='test',
        max_latex_len=max_latex_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
