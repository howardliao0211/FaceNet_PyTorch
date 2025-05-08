import torch.nn.functional as F
import torch

def semi_negative_triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Triplet loss function.
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    semi_hard_negative_mask = (pos_dist < neg_dist)

    loss = F.relu(pos_dist - neg_dist + margin)
    semi_hard_loss = loss[semi_hard_negative_mask]

    if semi_hard_loss.numel():
        return semi_hard_loss.mean()
    else:
        return None

def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Triplet loss function.
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def train_loop(model, dataloader, optimizer, loss_fn, margin=0.2, device='cpu'):
    """
    Perform one training loop over the dataset.
    """
    model.train()
    
    train_loss = 0.0

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
        if loss is None:
            continue
        
        # Backward pass & optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            train_loss += loss.item()
            index = (batch + 1) * dataloader.batch_size
            print(f'    loss: {loss.item(): 8.5f} ----- {index: 6d} / {len(dataloader.dataset)}')
    
    return train_loss / len(dataloader.dataset)

def test_loop(model, dataloader, loss_fn, margin=0.2, device='cpu', distance_threshold=1.1):
    model.eval()

    test_loss = 0.0
    true_accepts = 0.0
    false_accepts = 0.0

    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            batch_size = anchor.size(0)
            mini_batch = torch.cat((anchor, positive, negative), dim=0)
            embeddings = model(mini_batch)

            anchor_out = embeddings[:batch_size]
            positive_out = embeddings[batch_size:2*batch_size]
            negative_out = embeddings[2*batch_size:]

            loss = loss_fn(anchor_out, positive_out, negative_out, margin)
            test_loss += loss.item()
            
            # Calculate validation rate and false accept rate
            positive_distance = F.pairwise_distance(anchor_out, positive_out)
            true_accepts += (positive_distance < distance_threshold).sum().item()

            negative_distance = F.pairwise_distance(anchor_out, negative_out)
            false_accepts += (negative_distance < distance_threshold).sum().item()

    test_loss /= len(dataloader)
    val_rate = true_accepts / (len(dataloader) * dataloader.batch_size)
    far_rate = false_accepts / (len(dataloader) * dataloader.batch_size)
    print(f'Test loss: {test_loss: 8.3f}. Validation rate: {val_rate: 5.3f}, False accept rate: {far_rate: 5.3f}')

    return test_loss