import torchvision.transforms.functional as F
from config import IMAGE_SIZE

class ImagePreprocessing:
    def __init__(
        self, 
        target_size=IMAGE_SIZE, 
        fill=255,
        to_tensor=True,
        normalize=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ):
        self.target_size = target_size
        self.fill = fill
        self.to_tensor = to_tensor
        self.normalize = normalize
        self.mean = mean
        self.std = std
    
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
        
        if self.to_tensor:
            image = F.to_tensor(image)
        
        if self.normalize and self.to_tensor:
            image = F.normalize(image, mean=self.mean, std=self.std)
        
        return image
