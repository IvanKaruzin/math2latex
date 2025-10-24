import os
import torch
import cv2
import matplotlib.pyplot as plt

from dataset import Vocab, resize_and_pad


def test_inference(model, vocab, device="cuda", model_path=None, test_image_path=None, val_loader=None):
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model = model.to(device)
    model.eval()
    
    if test_image_path is None:
        if val_loader is None:
            raise ValueError("Either test_image_path or val_loader must be provided")
        # get a random sample from validation dataset
        val_dataset = val_loader.dataset
        random_idx = torch.randint(0, len(val_dataset), (1,)).item()
        img, true_tokens = val_dataset[random_idx]
        pad_id = vocab.stoi[vocab.pad]
        true_text = vocab.decode([t for t in true_tokens.tolist() if t != pad_id])
        img_np = img.squeeze(0).numpy()  # [H,W]
    else:
        if not os.path.exists(test_image_path):
            print(f"Файл {test_image_path} не найден")
            return
        
        img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Не удалось загрузить изображение {test_image_path}")
            return
        
        img = resize_and_pad(img)
        img_np = img.copy()  # сохраним для отображения
        img = img / 255.0
        img = torch.tensor(img).unsqueeze(0).float()
        true_text = None

    result_tokens = model.greedy_decode(img.to(device), vocab, device)
    result_text = vocab.decode([t for t in result_tokens if t != vocab.stoi[vocab.pad]])
    
    print(f"Результат: {result_text}")
    if true_text is not None:
        print(f"Истинная:  {true_text}")

    plt.imshow(img_np, cmap="gray")
    plt.axis("off")
    plt.title("Test Image")
    plt.show()
    
    return result_text


if __name__ == "__main__":
    print("Run this module by importing test_inference from your training script or notebook.")
