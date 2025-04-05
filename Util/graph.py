from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def show_triplet_img(dataset) -> None:
    """
    Function to show triplet images in a grid format.
    Args:
        dataset: List of triplet images (anchor, positive, negative)
    """
    num_triplets = len(dataset)
    fig, axes = plt.subplots(num_triplets, 3, figsize=(9, 3 * num_triplets))

    # If there's only one triplet, axes won't be a 2D array
    if num_triplets == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, data in enumerate(dataset):
        anchor, positive, negative = data

        # Convert tensors to numpy arrays for visualization
        anchor_img = anchor.permute(1, 2, 0).numpy() if isinstance(anchor, torch.Tensor) else anchor
        positive_img = positive.permute(1, 2, 0).numpy() if isinstance(positive, torch.Tensor) else positive
        negative_img = negative.permute(1, 2, 0).numpy() if isinstance(negative, torch.Tensor) else negative

        # Plot anchor, positive, and negative images
        axes[i, 0].imshow(anchor_img)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(positive_img)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(negative_img)
        axes[i, 2].axis("off")

    fig.suptitle("Anchor, Positive, Negative", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    anchor = torch.rand(3, 224, 224)  # Example tensor for anchor image
    positive = torch.rand(3, 224, 224)  # Example tensor for positive image
    negative = torch.rand(3, 224, 224)  # Example tensor for negative image
    dataset = [(anchor, positive, negative), (anchor, positive, negative), (anchor, positive, negative)]  # Example dataset with one triplet
    show_triplet_img(dataset)
