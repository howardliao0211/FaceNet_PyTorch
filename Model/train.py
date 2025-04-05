import torch.nn.functional as F
import torch

def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Triplet loss function.
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = torch.mean(F.relu(pos_dist - neg_dist + margin))
    return loss


def train_loop(model, dataloader, optimizer, loss_fn, margin=0.2):
    """
    Perform one training loop over the dataset.
    """
    model.train()
    
    losses = []
    for batch, (anchor, positive, negative) in enumerate(dataloader):
        # Forward pass
        anchor_loss, positive_loss, negative_loss = model(anchor), model(positive), model(negative)
        loss = loss_fn(anchor_loss, positive_loss, negative_loss, margin)
        
        # Backward pass & optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            losses.append(loss.item())
            index = (batch + 1) * dataloader.batch_size
            print(f'    loss: {loss.item(): 3.3f} ----- {index: 6d} / {len(dataloader.dataset)}')
    
    return losses
