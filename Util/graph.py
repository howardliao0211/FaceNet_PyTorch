from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def show_image_pairs(imgs) -> None:
    fig, axs = plt.subplots(nrows=len(imgs), ncols=2, figsize=(6, 3 * len(imgs)))
    
    for i, (img1, img2, issame) in enumerate(imgs):
        # Convert tensors to PIL images
        img1 = F.to_pil_image(img1.detach())
        img2 = F.to_pil_image(img2.detach())
        
        # Display the images
        axs[i, 0].imshow(np.asarray(img1))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
        axs[i, 1].imshow(np.asarray(img2))
        axs[i, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        # Add label above the pair
        label = "Same" if issame else "Different"
        axs[i, 0].set_title(label, fontsize=12)
    
    plt.tight_layout()
    plt.show()

