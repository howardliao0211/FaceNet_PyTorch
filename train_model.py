import torch.optim.adadelta
from Model import FaceNet, MiniFaceNet, train_loop, test_loop, semi_negative_triplet_loss
from Data.data import get_dataloader
from Trainers.trainers import graph_loss
from torch import nn
from torchvision import transforms
from Util.graph import show_triplet_img
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
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05)

    epochs = 50
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loop(model, train_loader, optimizer, semi_negative_triplet_loss, device=device)
        test_loop(model, test_loader, semi_negative_triplet_loss, device=device)

    torch.save(model, 'facenet_pytorch.pkl')
