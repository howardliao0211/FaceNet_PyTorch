from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import random
from collections import defaultdict
from pathlib import Path

class TripletDataset(datasets.ImageFolder):
    def __init__(self, dir: str, transform=None):

        super(TripletDataset, self).__init__(dir, transform)

        self.transform = transform
        self.label_dict = self.create_label_dict(dir)
        self.validation_images = self.create_triplet_dataset(self.label_dict)

    def create_label_dict(self, dir: str) -> dict[str, list]:
        """
        Create a dictionary of labels and their corresponding image paths.
        """

        label_dict = defaultdict(list)
        dataset_dir = Path(dir)

        for label in dataset_dir.iterdir():
            label_name = label.parts[-1]
            label_dict[label_name] = [files for files in label.iterdir() if files.is_file()]
        
        return label_dict
    
    def create_triplet_dataset(self, label_dict) -> list:
        """
        Create a triplet dataset from the label dictionary.
        Each entry in the dataset is a tuple of (anchor, positive, negative).
        """

        triplet_dataset = []

        for label, images in label_dict.items():

            # Need at least 2 images for anchor and positive
            while len(images) >= 2:

                # Randomly select an anchor and positive image from the same label
                anchor, positive = random.sample(images, 2)
                negative_label = random.choice([l for l in label_dict.keys() if l != label and len(label_dict[l]) > 0])
                negative = random.choice(label_dict[negative_label])

                triplet_dataset.append((anchor, positive, negative))

                images.remove(anchor)
                images.remove(positive)

        return triplet_dataset

    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img) if self.transform else img

        if isinstance(index, slice):
            res = []
            for anchor, positive, negative in self.validation_images[index]:
                res.append((transform(anchor), transform(positive), transform(negative)))
            return res
        else:
            anchor, positive, negative = self.validation_images[index]
            return transform(anchor), transform(positive), transform(negative)

    def __len__(self):
        return len(self.validation_images)

def get_dataloader(dir: str, transform=transforms.ToTensor(), val_percent=10, batch_size=32) -> Tuple[DataLoader, DataLoader]:
    dataset = TripletDataset(dir=dir, transform=transform)
    
    # Split the dataset into training and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    sample_size = int(dataset_size * (val_percent / 100))

    valid_dataset = Subset(dataset, indices[:sample_size])
    train_dataset = Subset(dataset, indices[sample_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader
