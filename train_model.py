from Model import FaceNet, MiniFaceNet, train_loop, triplet_loss
from Data.data import get_dataloader
from Trainers.trainers import graph_loss
import torch
from torch import nn

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
    model = MiniFaceNet()
    # model = FaceNet()

    model.to(device)
    train_loader, test_loader = get_dataloader(dir=LFW_DIR, batch_size=64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        print(f"Epoch {epoch+1}/{10}")
        train_loop(model, train_loader, optimizer, triplet_loss, device=device)