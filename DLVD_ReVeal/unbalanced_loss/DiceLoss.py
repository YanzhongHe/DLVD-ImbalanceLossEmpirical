import torch


def one_hot_encode(labels, num_classes):

    return torch.eye(num_classes)[labels].to(labels.device)
def dice_loss(logits, labels, smooth=1, num_classes=2):  # DL

    preds = torch.sigmoid(logits)

    labels = one_hot_encode(labels, num_classes=num_classes).float()

    preds = preds.float()

    inse = torch.sum(preds * labels, dim=1)

    l = torch.sum(preds * preds, dim=1)
    r = torch.sum(labels * labels, dim=1)

    dice = (2. * inse + smooth) / (l + r + smooth)

    loss = torch.mean(1 - dice)
    return loss
