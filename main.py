from Data.data import get_dataloader
from Util.graph import show_image_pairs
from torchvision import transforms
from torch.utils.data import DataLoader

LFW_DIR_PATH = r'Data\lfw_224'
PAIRS_PATH = r'Data\lfw_pairs.txt'

if __name__ == '__main__':
    train_loader, test_loader = get_dataloader(dir=LFW_DIR_PATH, pairs_path=PAIRS_PATH, transform=transforms.ToTensor())
    print(f"Train loader: {len(train_loader.dataset)}, Test loader: {len(test_loader.dataset)}")

    
