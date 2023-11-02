from PIL import Image
from torchvision.transforms import functional as F

def open_img(path):
    img = Image.open(path).convert("RGB")
    img = F.pil_to_tensor(img)
    # To float32
    img = img / 255.0
    img = F.resize(img, size=(299, 299), antialias=True)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img
