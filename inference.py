import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from Util import show_triplet_img
from Model import *
from Data import *

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
    LFW_DIR = r'./Data/lfw_224.zip'
    _, test_loader = get_dataloader(LFW_DIR, transform=None, val_percent=10, batch_size=64)

    model = FaceNet()
    checkpoint = torch.load(r'Checkpoints\20250411_115256\FaceNet_ResNeXt_epoch20.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data_index = 77
    dataset = test_loader.dataset

    anchor_image = dataset[data_index][0]
    positive_image = dataset[data_index][1]
    negative_image = dataset[data_index][2]

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
