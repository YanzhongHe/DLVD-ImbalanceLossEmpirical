import torch

def one_hot_encode(labels, num_classes):
    """将标签转换为独热编码"""
    return torch.eye(num_classes)[labels].to(labels.device)

def dice_loss(logits, labels, smooth=1, num_classes=2):  # DL
    # 将logits转换为概率（假设二元分类问题）
    preds = torch.sigmoid(logits)
    
    # 将labels转换为浮点数类型，并转换为独热编码
    labels = one_hot_encode(labels, num_classes=num_classes).float()
    
    # 确保输入是浮点数类型
    preds = preds.float()

    # 计算相交部分（交集）
    inse = torch.sum(preds * labels, dim=1)

    # 计算平方和
    l = torch.sum(preds * preds, dim=1)
    r = torch.sum(labels * labels, dim=1)

    # 计算Dice系数
    dice = (2. * inse + smooth) / (l + r + smooth)

    # 计算平均损失
    loss = torch.mean(1 - dice)
    return loss
