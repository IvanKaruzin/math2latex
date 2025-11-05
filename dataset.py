import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from datasets import load_dataset
from torchvision import transforms
from typing import Optional
import torchvision.transforms.functional as F
from build_vocabulary import LaTeXTokenizer

class ResizeWithPad:
    def __init__(self, target_size=224, fill=255):
        self.target_size = target_size
        self.fill = fill
    
    def __call__(self, image):
        w, h = image.size
        
        ratio = self.target_size / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        image = F.resize(image, (new_h, new_w), antialias=True)
        
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2)
        )
        
        image = F.pad(image, padding, fill=self.fill, padding_mode='constant')
        
        return image

class LaTeXDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        vocab,
        cache_dir: str = r'D:\datasets',
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        max_latex_len: int = 512
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
        return transforms.Compose([
            ResizeWithPad(target_size=224, fill=255),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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
    images_dir: str = r"D:\datasets\extraction\root\images",
    vocab=None,
    cache_dir: str = r'D:\datasets',
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
