import torchvision.transforms.functional as F
from config import IMAGE_SIZE

class ResizeWithPad:
    def __init__(self, target_size=IMAGE_SIZE, fill=255):
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