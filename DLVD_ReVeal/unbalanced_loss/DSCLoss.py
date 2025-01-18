import torch


def one_hot_encode(labels, num_classes):

    return torch.eye(num_classes)[labels].to(labels.device)


def DSC_loss(logits, labels, smooth=0.5, num_classes=2):

    probs = torch.sigmoid(logits)

    labels = one_hot_encode(labels, num_classes=num_classes)

    if probs.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: probs shape {probs.shape} and labels shape {labels.shape} should be the same")

    probs_rev = 1.0 - probs


    nominator = 2.0 * torch.sum(probs * labels, dim=1) + smooth
    denominator = torch.sum(probs * probs, dim=1) + torch.sum(labels * labels, dim=1) + smooth

    dice = nominator / denominator

    loss = 1 - dice

    return torch.mean(loss)
