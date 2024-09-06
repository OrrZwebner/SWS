import torch
import torch.nn.functional as F

PAD_TOKEN_LABEL = -100  # Special padding token for labels

def weighted_loss_function(logits, labels, weight):
    """
    Computes the weighted cross-entropy loss.

    Parameters:
    -----------
    logits : torch.Tensor
        The predicted logits from the model.
    labels : torch.Tensor
        The ground truth labels.
    weight : torch.Tensor
        The weights for each class.

    Returns:
    --------
    torch.Tensor
        The computed weighted cross-entropy loss.
    """
    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), weight=weight, ignore_index=PAD_TOKEN_LABEL)
    loss = F.cross_entropy(logits, labels, weight=weight, ignore_index=PAD_TOKEN_LABEL)
    return loss