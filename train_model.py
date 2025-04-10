import torch.optim.adadelta
from Model import *
from Data import get_dataloader
from Trainers.trainers import graph_loss
from torch import nn
from torchvision import transforms
from Util.graph import show_triplet_img
from pathlib import Path
import datetime
import torch

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128)
        )

    def forward(self, x):
        x = self.net(x)
        l2 = torch.sqrt(x**2)
        return l2

def get_model_file_path(dir: str, model_name: str, epoch=None) -> str:
    model_directory = Path(dir)
    model_directory.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y_%m_%d")

    if epoch:
        model_file_name = f'{model_name}_epoch{epoch}_{date}.pt'
    else:
        model_file_name = f'{model_name}_{date}.pt'

    return str(model_directory/model_file_name)

if __name__ == "__main__":
    LFW_DIR = r'./Data/lfw_224.zip'

    # Define your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = TestModel()
    # model = MiniFaceNet()
    model = FaceNet()

    model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_loader, test_loader = get_dataloader(dir=LFW_DIR, batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    model_dir = Path(r'./Checkpoints')
    model_name = 'FaceNet_ResNeXt'
    model_file_path = get_model_file_path(model_dir, model_name)

    epochs = 20
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loop(model, train_loader, optimizer, semi_negative_triplet_loss, device=device)
        test_loss, val_rate, far_rate = test_loop(model, test_loader, triplet_loss, device=device)
        
        # Record Check Point
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': test_loss,
            'val': val_rate,
            'far': far_rate,
        }, get_model_file_path(model_dir, model_name, epoch))

