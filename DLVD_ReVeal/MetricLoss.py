import torch

def Triplet_Loss(batch, selected_ids, protos, label, contrastive_loss, reg_alpha=0.0):
    """
        Certainty-Aware Triplet Loss
    """
    margin = 1.0
    num_contrastive_samples = 2
    variance = batch['variance']
    batch_size = batch['variance'].size(0)
    a_norm = torch.tensor(0.0).to(protos.device)
    variance = variance.to(protos.device)

    for ids in range(batch_size):

        if ids in selected_ids:
            continue

        current_proto = protos[ids]
        current_label = label[ids]

        positive_ids = (label == current_label)
        negative_ids = ~positive_ids

        # argsort yields ascending order
        selected_positive_ids = torch.argsort(variance[positive_ids])[:num_contrastive_samples]
        selected_negative_ids = torch.argsort(variance[negative_ids])[:num_contrastive_samples]

        positive_samples = protos[positive_ids][selected_positive_ids]
        negative_samples = protos[negative_ids][selected_negative_ids]
        positive_distance = torch.mean(torch.square(current_proto - positive_samples), dim=1).sum()
        negative_distance = torch.mean(torch.square(current_proto - negative_samples), dim=1)
        negative_distance = torch.square(torch.clamp(margin - torch.sqrt(negative_distance), min=0)).sum()

        # Up sampling
        if len(selected_positive_ids) != 0:
            positive_distance = positive_distance / len(selected_positive_ids)
        if len(selected_negative_ids) != 0:
            negative_distance = negative_distance / len(selected_negative_ids)

        contrastive_loss += (positive_distance + negative_distance) / 2
        # reg
        a_norm += torch.norm(current_proto, dim=-1)

    a_norm = a_norm / batch_size
    contrastive_loss = contrastive_loss / batch_size

    return contrastive_loss + reg_alpha * a_norm
