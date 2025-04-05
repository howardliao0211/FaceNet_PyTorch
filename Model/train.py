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


def train_loop(model, dataloader, optimizer, loss_fn, margin=0.2, device='cpu'):
    """
    Perform one training loop over the dataset.
    """
    model.train()
    
    losses = []
    for batch, (anchor, positive, negative) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        mini_batch = torch.cat((anchor, positive, negative), dim=0)
        embeddings = model(mini_batch)  # shape: [3 * batch_size, embedding_dim]

        # Split the embeddings back
        batch_size = anchor.size(0)
        anchor_out = embeddings[:batch_size]
        positive_out = embeddings[batch_size:2*batch_size]
        negative_out = embeddings[2*batch_size:]

        loss = loss_fn(anchor_out, positive_out, negative_out, margin)
        
        # Backward pass & optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            losses.append(loss.item())
            index = (batch + 1) * dataloader.batch_size
            print(f'    loss: {loss.item(): 8.3f} ----- {index: 6d} / {len(dataloader.dataset)}')
    
    return losses
