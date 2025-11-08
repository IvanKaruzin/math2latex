import torch
from vocab import Vocab
from dataset import get_dataloaders
from model import FormulaRecognizer
from train import train_model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab = Vocab(vocab_path='tokens.json')
    
    train_loader, val_loader, test_loader = get_dataloaders(
        images_dir=r"D:\datasets\extraction\root\images",
        vocab=vocab,
        cache_dir=r'D:\datasets',
        batch_size=128,
        num_workers=0,
        max_latex_len=80
    )
    
    model = FormulaRecognizer(vocab_size=len(vocab), hidden_dim=256)
    
    test_loss, test_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tokenizer=vocab,
        epochs=2,
        device=device,
        resume=True
    )
