from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

HF_DATASET_NAME = "hoang-quoc-trung/fusion-image-to-latex-datasets"

IMAGES_DIR = Path(r"D:\datasets\extraction\root\images")
HF_CACHE_DIR = Path(r"D:\datasets")

VOCAB_PATH = BASE_DIR / "tokens.json"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint_last.pth"
BEST_MODEL_PATH = CHECKPOINT_DIR / "checkpoint_best.pth"

IMAGE_SIZE = 224

MAX_LATEX_LEN = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
