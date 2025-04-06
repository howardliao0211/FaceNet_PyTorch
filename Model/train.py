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
            print(f'    loss: {loss.item(): 8.5f} ----- {index: 6d} / {len(dataloader.dataset)}')
    
    return losses

def test_loop(model, dataloader, loss_fn, margin=0.2, device='cpu'):
    model.eval()

    test_loss = 0.0
    anchor_emb = 0.0
    positive_emb = 0.0
    negative_emb = 0.0

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
            anchor_emb += anchor_out.mean().item()
            positive_emb += positive_out.mean().item()
            negative_emb += negative_out.mean().item()

    test_loss /= len(dataloader)
    anchor_emb /= len(dataloader)
    positive_emb /= len(dataloader)
    negative_emb /= len(dataloader)
    print(f'Test loss: {test_loss: 8.5f}. anchor: {anchor_emb: 8.5f}, positive: {positive_emb: 8.5f}, negative: {negative_emb: 8.5f}')
