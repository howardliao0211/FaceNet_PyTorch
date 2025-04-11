import torch.optim.adadelta
from Model import *
from Data import get_dataloader
from Trainers.trainers import graph_loss
from torch import nn
from torchvision import transforms
from Util.graph import show_triplet_img
from pathlib import Path
from typing import Any
import argparse
import datetime
import torch

def get_checkpoint_dir(dir: str) -> str:
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_directory = Path(dir) / date
    model_directory.mkdir(parents=True, exist_ok=True)

    print(f'Checkpoint Directory: {str(model_directory)}')
    return str(model_directory)

def get_checkpoint_path(dir: str, model_name: str, epoch=None) -> str:
    model_directory = Path(dir)
    model_directory.mkdir(parents=True, exist_ok=True)

    if epoch:
        model_file_name = f'{model_name}_epoch{epoch}.pt'
    else:
        model_file_name = f'{model_name}.pt'

    return str(model_directory/model_file_name)

if __name__ == "__main__":
    # Create parser object to parse user argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    # The filepath of the dataset. Consists of ~4000 dataset of anchor, positive, and negative. 
    LFW_DIR = r'./Data/lfw_224.zip'

    # Get the device of the current environment.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Current device: {device}')

    # Create model, train loader, test loader, and optimizer. 
    model = FaceNet()
    model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_loader, test_loader = get_dataloader(dir=LFW_DIR, batch_size=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Load checkpoint if given. 
    if args.load_checkpoint:
        checkpoint: dict[str, Any] = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')

        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        print(f"Checkpoint loaded from {args.load_checkpoint}")
        for key, value in checkpoint.items():
            if key.endswith('state_dict'):
                continue
            print(f'{key}: {value}')
    else:
        print("No checkpoint specified. Initializing model from scratch.")
    
    # Load learning rate for the optimizer. 
    if args.lr:
        print(f'Loading learning rate from argument. lr={args.lr}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    # Actual Training. 
    model_name = 'FaceNet_ResNeXt'
    checkpoint_dir = get_checkpoint_dir(r'./Checkpoints')

    epochs = args.epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loop(model, train_loader, optimizer, semi_negative_triplet_loss, device=device)
        test_loss, val_rate, far_rate = test_loop(model, test_loader, triplet_loss, device=device)
        
        # Record Check Point
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
            'val': val_rate,
            'far': far_rate,
        }, get_checkpoint_path(checkpoint_dir, model_name, epoch + 1))

