import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
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

def is_same_face(model: torch.nn.Module, image1: torch.Tensor, image2: torch.Tensor, threshold: float) -> bool:
    image1, image2 = preprocess_image(image1), preprocess_image(image2)

    with torch.no_grad():
        embedding1, embedding2 = model(image1), model(image2)
    
    return postprocess_embedding(embedding1, embedding2) < threshold

if __name__ == "__main__":
    
    # Load Model. 
    model_path = r"Trained_Models\Baseline_FaceNet.pkl"
    LFW_DIR = r'./Data/lfw_224.zip'
    _, test_loader = get_dataloader(LFW_DIR, transform=None, val_percent=10, batch_size=64)

    model = FaceNet()
    checkpoint: dict[str] = torch.load(r'Checkpoints\20250411_115256\FaceNet_ResNeXt_epoch20.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print('Model Checkpoint Statistic: ')
    for k, v in checkpoint.items():
        if k.endswith('state_dict'):
            continue
        print(f'{k:5s}: {v}')

    data_to_inference = 5
    dataset = test_loader.dataset

    test_images: list[tuple] = []
    test_answer = []

    # Create inference dataset. 
    for i in range(data_to_inference):
        choice = random.choice([1, 2]) # index 1 is positive and index 2 is negative. 
        test_images.append((dataset[i][0], dataset[i][choice]))
        test_answer.append(choice == 1)
    
    # Create prediction. 
    embedding_threshold = 1.1
    test_pred = []
    for test in test_images:
        pred = is_same_face(model, test[0], test[1], embedding_threshold)
        test_pred.append(pred)

    # Plot results
    plt.figure()

    for i in range(data_to_inference):
        
        plt.subplot(data_to_inference, 2, 2*i + 1)
        plt.imshow(test_images[i][0])
        plt.axis('off')
        plt.title(f'Answer: {test_answer[i]}')
        
        plt.subplot(data_to_inference, 2, 2*i + 2)
        plt.imshow(test_images[i][1])
        plt.axis('off')
        plt.title(f'Pred: {test_pred[i]}')

    plt.tight_layout()
    plt.show()

