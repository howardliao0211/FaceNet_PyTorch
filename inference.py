import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from Util import show_triplet_img
from Model import *

def load_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)
    
    return image

def postprocess_embedding(embeddging1: torch.Tensor, embedding2: torch.Tensor) -> float:
    x = F.pairwise_distance(embeddging1, embedding2)
    x = x.item()
    return x

if __name__ == "__main__":
    model_path = r"Trained_Models\Baseline_FaceNet.pkl"
    anchor_path = r"Data\lfw_224\Adam_Sandler\Adam_Sandler_0001.jpg"
    positive_path = r"Data\lfw_224\Adam_Sandler\Adam_Sandler_0003.jpg"
    negative_path = r"Data\lfw_224\Abdel_Madi_Shabneh\Abdel_Madi_Shabneh_0001.jpg"

    model = FaceNet()
    checkpoint = torch.load(r'Checkpoints\FaceNet_ResNeXt_20250410_170040_epoch16.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    anchor_image = load_image(anchor_path)
    positive_image = load_image(positive_path)
    negative_image = load_image(negative_path)

    distance_threshold = 1.1

    with torch.no_grad():
        anchor_embedding = model(preprocess_image(anchor_image))
        positive_embedding = model(preprocess_image(positive_image))
        negative_embedding = model(preprocess_image(negative_image))

        anchor_to_positive_distance = postprocess_embedding(anchor_embedding, positive_embedding)
        anchor_to_negative_distance = postprocess_embedding(anchor_embedding, negative_embedding)
    
        print(f"Distance between anchor and positive: {anchor_to_positive_distance}")
        print(f"Distance between anchor and negative: {anchor_to_negative_distance}")
    
    show_triplet_img([(anchor_image, positive_image, negative_image)])


