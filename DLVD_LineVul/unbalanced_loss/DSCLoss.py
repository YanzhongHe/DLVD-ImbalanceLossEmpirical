import torch

def one_hot_encode(labels, num_classes):
    # 将标签转换为独热编码
    return torch.eye(num_classes)[labels].to(labels.device)



def DSC_loss(logits, labels, smooth=0.5, num_classes=2):
    # 将logits转换为概率
    probs = torch.sigmoid(logits)  # 对于二分类问题
    
    # 转换标签为独热编码
    labels = one_hot_encode(labels, num_classes=num_classes)
    
    # 确保形状一致
    if probs.shape != labels.shape:
        raise ValueError(f"Shape mismatch: probs shape {probs.shape} and labels shape {labels.shape} should be the same")
    
    # 计算1减去预测值
    probs_rev = 1.0 - probs
    
    # 计算交集和平方和
    nominator = 2.0 * torch.sum(probs * labels, dim=1) + smooth
    denominator = torch.sum(probs * probs, dim=1) + torch.sum(labels * labels, dim=1) + smooth
    
    # 计算Dice系数
    dice = nominator / denominator
    
    # 计算DSC损失
    loss = 1 - dice
    
    # 返回损失的均值
    return torch.mean(loss)
