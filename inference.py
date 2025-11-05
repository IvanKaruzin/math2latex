import torch
import matplotlib.pyplot as plt
import random
from vocab import Vocab
from dataset import LaTeXDataset
from model import FormulaRecognizer

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def test_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vocab = Vocab(vocab_path='tokens.json')
    
    dataset = LaTeXDataset(
        images_dir=r"D:\datasets\extraction\root\images",
        vocab=vocab,
        cache_dir=r'D:\datasets',
        split='test',
        max_latex_len=64
    )
    
    model = FormulaRecognizer(vocab_size=len(vocab), hidden_dim=256)
    
    checkpoint = torch.load('model_best.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from checkpoint (epoch {checkpoint['epoch']})")
    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    num_samples = 5
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, tokens = dataset[idx]
            
            img_denorm = denormalize_image(image)
            img_np = img_denorm.permute(1, 2, 0).numpy()
            
            ground_truth = vocab.decode(tokens.tolist())
            
            predicted_ids = model.greedy_decode(image, vocab, device=device)
            predicted_latex = vocab.decode(predicted_ids)
            
            axes[i].imshow(img_np)
            axes[i].set_title(
                f"Ground Truth: {ground_truth[:80]}...\n"
                f"Predicted: {predicted_latex[:80]}...",
                fontsize=10,
                wrap=True
            )
            axes[i].axis('off')
            
            print(f"\nSample {i+1}:")
            print(f"Ground Truth: {ground_truth}")
            print(f"Predicted:    {predicted_latex}")
            print(f"Match: {ground_truth == predicted_latex}")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_inference()
