import torch.optim.adadelta
from Model import *
from Data import get_dataloader
from torch import nn
from torchvision import transforms
from Util.graph import show_triplet_img
from pathlib import Path
from typing import Any
from trainers.core import BaseTrainer
import argparse
import datetime
import torch

class FacenetTrainer(BaseTrainer):

    def train_loop(self):
        return train_loop(
            model=self.model,
            dataloader=self.train_loader,
            optimizer=self.optimizer,
            loss_fn=semi_negative_triplet_loss,
            device=self.device
        )

    def test_loop(self):
        return test_loop(
            model=self.model,
            dataloader=self.test_loader,
            loss_fn=triplet_loss,
            device=self.device
        )

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
    trained_epoch = 0
    if args.load_checkpoint:
        checkpoint: dict[str, Any] = torch.load(args.load_checkpoint, map_location=device)
        trained_epoch = checkpoint['epoch']
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
    trainer = FacenetTrainer(
        name='Facenet',
        model=model,
        optimizer=optimizer,
        loss_fn=None, # train and test loop have different loss function. 
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )

    trainer.fit(args.epochs, trained_epochs=trained_epoch, graph=True, save_check_point=False)
