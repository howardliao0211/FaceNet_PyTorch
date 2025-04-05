from Model import FaceNet, MiniFaceNet, train_loop, triplet_loss
from Data.data import get_dataloader
from Trainers.trainers import graph_loss
import torch

if __name__ == "__main__":
    model = FaceNet()
    train_loader, test_loader = get_dataloader(dir=r'Data\lfw_224')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = train_loop(model, train_loader, optimizer, triplet_loss)
    graph_loss({'Train Loss': losses})