import torch

def one_hot_encode(labels, num_classes):
    return torch.eye(num_classes)[labels].to(labels.device)

def DSC_loss(logits, labels, gamma=1.0, smooth=1, num_classes=2):
    probs = torch.sigmoid(logits)  # Convert logits to probabilities
    labels = one_hot_encode(labels, num_classes=num_classes)  # One-hot encode ground truth

    if probs.shape != labels.shape:
        raise ValueError(f"Shape mismatch: probs shape {probs.shape} and labels shape {labels.shape} should be the same")

    one_minus_probs = 1.0 - probs
    weight = one_minus_probs ** gamma

    numerator = weight * probs * labels

    l = weight * probs
    r = labels

    numerator_sum = 2.0 *torch.sum(numerator, dim=1) + smooth
    denominator_sum = torch.sum(l, dim=1) + torch.sum(r, dim=1) + smooth

    loss = 1 - numerator_sum / denominator_sum
    return torch.mean(loss)