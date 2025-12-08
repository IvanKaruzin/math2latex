import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
})
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from vocab import Vocab
from preprosess_image import ImagePreprocessing
from model import FormulaRecognizer
from torchvision import transforms
from config import DEVICE, BEST_MODEL_PATH, IMAGE_SIZE

class FormulaProcessor:
    def __init__(self, checkpoint_path=BEST_MODEL_PATH, vocab_path='tokens.json'):
        self.device = DEVICE
        self.vocab = Vocab(vocab_path=vocab_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        vocab_size = checkpoint.get('vocab_size', len(self.vocab))
        self.model = FormulaRecognizer(vocab_size=vocab_size, hidden_dim=256)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = ImagePreprocessing(target_size=IMAGE_SIZE, fill=255)


    def recognize_formula(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        with torch.no_grad():
            predicted_ids = self.model.beam_search(image_tensor, self.vocab, device=self.device)
            predicted_latex = self.vocab.decode(predicted_ids)
        return predicted_latex

    def render_latex_to_image(self, latex_sequence):
        try:
            figure = plt.figure(figsize=(12, 5), dpi=200)
            figure.patch.set_alpha(0.0)
            ax = figure.add_subplot(111)
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                rf"$\displaystyle {latex_sequence}$",
                fontsize=60,
                ha="center",
                va="center"
            )
            buffer = BytesIO()
            plt.savefig(
                buffer,
                format="png",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.3
            )
            plt.close(figure)
            buffer.seek(0)
            return buffer
        except Exception as exc:
            raise RuntimeError(f"Ошибка компиляции LaTeX.")

def recognize_formula(image_path, checkpoint_path=BEST_MODEL_PATH, vocab_path='tokens.json'):
    device = DEVICE
    vocab = Vocab(vocab_path=vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = checkpoint.get('vocab_size', len(vocab))
    model = FormulaRecognizer(vocab_size=vocab_size, hidden_dim=256)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    transform = ImagePreprocessing(target_size=IMAGE_SIZE, fill=255)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    with torch.no_grad():
        predicted_ids = model.beam_search(image_tensor, vocab, device=device)
        predicted_latex = vocab.decode(predicted_ids)
    return predicted_latex