
import torch
from torchvision import transforms
from PIL import Image
from model import TransformerNet
import torchvision.transforms.functional as TF
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_path, max_size=512):
    image = Image.open(image_path).convert("RGB")
    size = max(image.size)
    if size > max_size:
        size = max_size
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def save_image(tensor, path):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image / 255
    image = TF.to_pil_image(image.clamp(0, 1))
    image.save(path)

def stylize_image(content_path, output_path, style_model_name):
    model_path = os.path.join("saved_models", style_model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {style_model_name} not found.")
    model = TransformerNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    content_image = image_loader(content_path)
    with torch.no_grad():
        output = model(content_image).cpu()
    save_image(output, output_path)
