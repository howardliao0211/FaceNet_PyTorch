from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import random
from collections import defaultdict
from pathlib import Path
import os
import zipfile

class TripletDataset(datasets.ImageFolder):
    def __init__(self, dir: str, transform=None):

        if dir.endswith('.zip'):
            if not os.path.exists(dir[:-4]):
                dir = self.extract_zip(dir)
            else:
                dir = dir[:-4]  # Remove .zip extension for folder name
        elif not os.path.isdir(dir):
            raise ValueError(f"Directory {dir} is not a zip or directory.")

        super(TripletDataset, self).__init__(dir)

        self.transform = transform
        self.label_dict = self.create_label_dict(dir)
        self.validation_images = self.create_triplet_dataset(self.label_dict)

    def extract_zip(self, zip_path: str) -> str:
        """
        Extracts a zip file to a temporary directory and returns the path to the extracted folder.
        """
        extracted_dir = Path(zip_path).with_suffix('')  # Remove .zip extension for folder name
        if not extracted_dir.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir.parent)
        return str(extracted_dir)
    
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
            
            # Create a copy of the images list to avoid modifying the original list
            cur_images = images[:]

            # Need at least 2 images for anchor and positive
            while len(cur_images) >= 2:

                # Randomly select an anchor and positive image from the same label
                anchor, positive = random.sample(cur_images, 2)
                negative_label = random.choice([l for l in label_dict.keys() if l != label and len(label_dict[l]) > 0])
                negative = random.choice(label_dict[negative_label])

                triplet_dataset.append((anchor, positive, negative))

                cur_images.remove(positive)

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

def get_dataloader(dir: str, transform, val_percent=10, batch_size=32) -> Tuple[DataLoader, DataLoader]:
    dataset = TripletDataset(dir=dir, transform=transform)
    
    # Split the dataset into training and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    sample_size = int(dataset_size * (val_percent / 100))

    valid_dataset = Subset(dataset, indices[:sample_size])
    train_dataset = Subset(dataset, indices[sample_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader
