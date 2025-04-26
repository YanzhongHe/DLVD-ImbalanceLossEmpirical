import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from stopper import EarlyStopping
from logger import TrainingLogger, MetricMeters
from .dataset import GraphDataset, collate_fn
from .model import ReVeal
import yaml
from transformers import logging, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, f1_score, recall_score, matthews_corrcoef, cohen_kappa_score, \
    roc_auc_score, confusion_matrix
# from balanced_loss import Loss
from .bigvul_dataset import BigVulGraphDataset

from unbalanced_loss.LDAMLoss import LDAMLoss
from unbalanced_loss.weight_ce_loss import WBCEWithLogitLoss
from unbalanced_loss.LALoss import LALoss
from unbalanced_loss.class_balanced_loss import Loss
from unbalanced_loss.GHMC import GHMC

from unbalanced_loss.DSCLoss import DSC_loss
from unbalanced_loss.DiceLoss import dice_loss


def train_teacher(ckpt_path, config, seed_s):
    train_config = config['train']
    data_config = config['data']
    loss_config = config['loss']
    # Set random seed
    set_random_seed(train_config['random_seed'])

    # Get data loader
    if 'big-vul' in data_config["ssl_data_path"]:
        val_dset = BigVulGraphDataset(f'{data_config["ssl_data_path"]}/output/seed{seed_s}/combined_val')
        test_dset = BigVulGraphDataset(f'{data_config["ssl_data_path"]}/output/seed{seed_s}/combined_test')
        train_dset = BigVulGraphDataset(f'{data_config["ssl_data_path"]}/output/seed{seed_s}/combined_train',
                                        mode='train')
    else:
        val_dset = GraphDataset(f'{data_config["ssl_data_path"]}/seed{seed_s}/val_split.pkl')
        test_dset = GraphDataset(f'{data_config["ssl_data_path"]}/seed{seed_s}/test_split.pkl')
        train_dset = GraphDataset(f'{data_config["ssl_data_path"]}/seed{seed_s}/train_split.pkl', mode='train')

    num_classes = train_dset.num_classes

    train_loader = DataLoader(train_dset, batch_size=train_config['batch_size'], shuffle=False, pin_memory=True,
                              num_workers=1, collate_fn=collate_fn)
    num_vul, num_none = 0, 0
    for data in train_loader:
        num_vul += sum(data['label'] == 1)
        num_none += sum(data['label'] == 0)
    samples_per_class = [num_none, num_vul]
    val_loader = DataLoader(val_dset, batch_size=train_config['eval_batch_size'], pin_memory=False, num_workers=1,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dset, batch_size=train_config['eval_batch_size'], pin_memory=False, num_workers=1,
                             collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ReVeal(input_channels=train_config['input_channels'], hidden_channels=train_config['hidden_channels'],
                   num_layers=train_config['num_layers'])
    max_steps = train_config['epoch_stop_patience'] * len(train_loader) * 3
    warmup_steps = len(train_loader) * train_config['epoch_stop_patience'] // 5
    optimizer, scheduler = get_optimizer(model, train_config, max_steps, warmup_steps)

    logger = TrainingLogger(dest_path=os.path.join(ckpt_path, f'log.txt'))

    logger.print('Number of classes: {}'.format(num_classes))
    logger.print('Length of train/val/test datasets: {}, {}, {}'.format(
        len(train_dset), len(val_dset), len(test_dset)))
    logger.print('Length of train/val/test dataloaders: {}, {}, {}'.format(
        len(train_loader), len(val_loader), len(test_loader)))
    logger.print(f'Using {torch.cuda.device_count()} GPUs: '
                 + ', '.join([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    logger.print('Config:\n' + str(config))
    logger.print('\n')

    # ------------- Start Training -------------#
    model = model.to(device)
    metric_meter = MetricMeters()
    train_steps = 0
    epoch_early_stopper = EarlyStopping(model, patience=train_config['epoch_stop_patience'], print_fn=logger.print,
                                        mode=train_config['stopper_mode'])

    while True:

        # Training
        metric_meter.reset(mode='train')
        train(train_loader, model, optimizer, scheduler, metric_meter, device, loss_config, samples_per_class)
        train_steps += 1
        scores = metric_meter.epoch_get_scores()
        logger.print('Steps {:2d} | Train Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Train Acc: {:5f}%, prec: {:5f}%, '
                     'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%'.format(
            train_steps, scores['total_loss'], scores['class_loss'], scores['metric_loss'], scores['accuracy'] * 100.0,
            scores['precision'], scores['recall'], scores['f1'], scores['mcc']))
        if train_steps >= 0:
            metric_meter.reset(mode='val')
            val(val_loader, model, metric_meter, device, loss_config, flag="val", ckpt_path=ckpt_path)
            scores = metric_meter.epoch_get_scores()
            logger.print(
                'Steps {:2d} | Val Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Val Acc: {:5f}%, prec: {:5f}%, '
                'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%, kappa: {:5f}%, auc: {:5f}%'.format(
                    train_steps, scores['total_loss'], scores['class_loss'], scores['metric_loss'],
                    scores['accuracy'] * 100.0,
                    scores['precision'], scores['recall'], scores['f1'], scores['mcc'], scores['kappa'], scores['auc']))
            save_ckpt = epoch_early_stopper(scores['mcc'], scores['class_loss'], scores['f1'], scores['accuracy'],
                                            scores['recall'])

            if save_ckpt:
                save_checkpoint(model, path=os.path.join(ckpt_path, 'best_model.pth'))

                if train_config['stopper_mode'] == 'f1':
                    logger.print(f'Best f1 = {scores["f1"]:.5f} achieved.Checkpoint saved.')
                elif train_config['stopper_mode'] == 'acc':
                    logger.print(f'Best acc = {scores["accuracy"]:.5f} achieved.Checkpoint saved.')
                elif train_config['stopper_mode'] == 'f1acc':
                    logger.print(
                        f'Best acc = {scores["accuracy"]:.5f} f1 = {scores["f1"]:.5f} achieved.Checkpoint saved.')
            if epoch_early_stopper.early_stop:
                break
            logger.print('\n')

    model_path = os.path.join(ckpt_path, 'best_model.pth')
    if not os.path.exists(model_path):
        return None
    load_checkpoint(model, model_path)

    metric_meter.reset(mode='test')
    confusion = val(test_loader, model, metric_meter, device, loss_config, flag="test", ckpt_path=ckpt_path)
    scores = metric_meter.epoch_get_scores()
    logger.print(
        'Test Loss(total/cls/me): {:5f}, {:5f}, {:5f} | Test Acc: {:5f}%, prec: {:5f}%, '
        'recall: {:5f}%, f1: {:5f}%, mcc: {:5f}%, kappa: {:5f}%, auc: {:5f}%, TN: {:d}, FP: {:d}, FN: {:d}, TP: {:d}'.format(
            scores['total_loss'], scores['class_loss'], scores['metric_loss'],
            scores['accuracy'] * 100.0,
            scores['precision'], scores['recall'], scores['f1'], scores['mcc'], scores['kappa'], scores['auc'],
            confusion[0][0], confusion[0][1], confusion[1][0], confusion[1][1]))
    metric_meter.dump(path=os.path.join(ckpt_path, f'teacher_metric_log.json'))
    os.rename(
        os.path.join(ckpt_path, 'best_model.pth'),
        os.path.join(ckpt_path, f'best_model.pth')
    )
    logger.print('\n' * 5)
    print("#######")
    print(ckpt_path)
    print("#######")

    return scores


def set_random_seed(seed):
    # print('Random seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(train_loader, model, optimizer, scheduler, metric_meter, device, config, samples_per_class):
    model.train()
    for i, batch in enumerate(train_loader, start=1):

        input_ids = batch['input_ids'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_type = batch['edge_type'].to(device)

        label = batch['label'].to(device)

        output = model(input_ids, edge_index, edge_type)
        logits = output['logits']
        labels_np = label.cpu().detach().numpy()
        output_np = logits.cpu().argmax(1).detach().numpy()

        # print(samples_per_class)

        negative_samples = samples_per_class[0]
        positive_samples = samples_per_class[1]

        total_samples = sum(samples_per_class)

        negative_ratio = negative_samples / total_samples
        positive_ratio = positive_samples / total_samples

        # loss function
        if config['loss_type'] == 'focalloss':
            loss_fct = Loss(
                loss_type="focal_loss_0",
                beta=0.999,  # class-balanced loss beta
                fl_gamma=2,  # focal loss gamma
                samples_per_class=samples_per_class,
                class_balanced=False
            )
            class_loss = loss_fct(logits, label)


        elif config['loss_type'] == 'celoss':
            class_loss = F.cross_entropy(logits, label, reduction='none')

        elif config['loss_type'] == 'CBfocalloss':
            loss_fct = Loss(
                loss_type="focal_loss",
                beta=0.999,  # class-balanced loss beta
                fl_gamma=2,  # focal loss gamma
                samples_per_class=samples_per_class,
                class_balanced=True
            )
            class_loss = loss_fct(logits, label)

        elif config['loss_type'] == 'CBceloss':
            loss_fct = Loss(
                loss_type="binary_cross_entropy",
                beta=0.999,  # class-balanced loss beta
                fl_gamma=2,  # focal loss gamma
                samples_per_class=samples_per_class,
                class_balanced=True
            )
            class_loss = loss_fct(logits, label)


        elif config['loss_type'] == 'GHMloss':

            neg_weight = 1.0 / negative_ratio
            pos_weight = 1.0 / positive_ratio

            batch_size = logits.size(0)
            label_weight = torch.zeros((batch_size, 2), device=logits.device)

            label_weight[:, 0] = neg_weight
            label_weight[:, 1] = pos_weight

            class_num = 2

            expanded_labels = torch.zeros((label.size(0), class_num), device=label.device)
            expanded_labels[range(len(label)), label] = 1

            loss_fct = GHMC(bins=10, momentum=0.9, use_sigmoid=True, loss_weight=1.0)
            class_loss = loss_fct(pred=logits, target=expanded_labels, label_weight=label_weight)


        elif config['loss_type'] == 'LDAMloss':
            loss_fct = LDAMLoss(cls_num_list=samples_per_class)
            class_loss = loss_fct(logits, label)

        elif config['loss_type'] == 'LAloss':
            cls_num_list = np.array(samples_per_class)
            cls_num_list = torch.from_numpy(cls_num_list)
            loss_fct = LALoss(cls_num_list=cls_num_list)
            class_loss = loss_fct(logits, label)

        elif config['loss_type'] == 'WCEloss':
            negative_weight = 1.0 / negative_samples
            positive_weight = 1.0 / positive_samples

            class_weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float).to(device)

            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            class_loss = loss_fct(logits, label)

        elif config['loss_type'] == 'DiceLoss':
            loss_fct = dice_loss
            class_loss = loss_fct(logits, label)

        elif config['loss_type'] == 'DSCLoss':
            loss_fct = DSC_loss
            class_loss = loss_fct(logits, label)

        class_loss = class_loss.mean()
        # Calculate metric loss
        metric_loss = torch.tensor(0.0).to(device)
        protos = output['hidden_state']
        unique_label = torch.unique(label)
        if config['coef_teacher'] == 0.0:
            protos = protos.detach()

        for l in unique_label:
            target_protos = protos[label == l]  # (-1, hidden size)
            centroid = torch.mean(target_protos, axis=0)  # (hidden_size)
            distance = torch.sum(((target_protos - centroid) ** 2), axis=1)
            metric_loss += torch.mean(distance, axis=0)
        metric_loss = metric_loss / len(unique_label)  # / protos.size(-1)
        optimizer.zero_grad()
        loss = class_loss + config['coef_teacher'] * metric_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        metric_meter.update('total_loss', loss.item(), total=1)
        metric_meter.update('class_loss', class_loss.item(), total=1)
        metric_meter.update('metric_loss', metric_loss.item(), total=1)
        metric_meter.update('accuracy',
                            correct=(logits.argmax(1) == label).sum().item(), total=len(label))
        metric_meter.update('precision',
                            correct=precision_score(labels_np, output_np), total=1)
        metric_meter.update('recall',
                            correct=recall_score(labels_np, output_np), total=1)
        metric_meter.update('f1',
                            correct=f1_score(labels_np, output_np), total=1)
        metric_meter.update('mcc',
                            correct=matthews_corrcoef(labels_np, output_np), total=1)


def val(val_loader, model, metric_meter, device, config, flag, ckpt_path):
    model.eval()
    confusion = None
    logit_list = []
    target_list = []
    predict_list = []
    flags = flag
    with torch.no_grad():
        for i, batch in enumerate(val_loader, start=1):

            input_ids = batch['input_ids'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_type = batch['edge_type'].to(device)
            label = batch['label'].to(device)

            output = model(input_ids, edge_index, edge_type)
            logits = output['logits']
            labels_np = label.cpu().detach().numpy()
            output_np = logits.cpu().argmax(1).detach().numpy()
            logit_list.append(F.softmax(logits).cpu().detach().numpy())
            target_list.append(labels_np)
            predict_list.append(output_np)

            ### Calculate class loss
            class_loss = F.cross_entropy(logits, label)

            ### Calculate metric loss
            metric_loss = torch.tensor(0.0).to(device)
            protos = output['hidden_state']
            unique_label = torch.unique(label)
            if config['coef_teacher'] == 0.0:
                protos = protos.detach()

            for l in unique_label:
                target_protos = protos[label == l]  # (-1, hidden size)
                centroid = torch.mean(target_protos, axis=0)  # (hidden_size)
                distance = torch.sum(((target_protos - centroid) ** 2), axis=1)
                metric_loss += torch.mean(distance, axis=0)
            metric_loss = metric_loss / protos.size(-1)

            loss = class_loss + config['coef_teacher'] * metric_loss
            if confusion is None:
                confusion = confusion_matrix(labels_np, output_np)
            else:
                confusion += confusion_matrix(labels_np, output_np)

            metric_meter.update('total_loss', loss.item(), total=1)
            metric_meter.update('class_loss', class_loss.item(), total=1)
            metric_meter.update('metric_loss', metric_loss.item(), total=1)
            metric_meter.update('accuracy',
                                correct=(logits.argmax(1) == label).sum().item(), total=len(label))

    logit_list = np.concatenate(logit_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    predict_list = np.concatenate(predict_list, axis=0)

    if flags == "test":
        df = pd.DataFrame({'True_Label': target_list, 'Predicted_Label': predict_list})

        comment = config['comment']

        csv_filename = f'{comment}.csv'

        csv_path = os.path.join(ckpt_path, csv_filename)
        print("#######")
        print(csv_path)
        print("#######")

        df.to_csv(csv_path, index=False)

        print(f'The CSV file has been saved in the directory {ckpt_path} ，The file name is {csv_filename}')

    confusion = confusion_matrix(target_list, predict_list)
    f1 = f1_score(target_list, predict_list)
    precision = precision_score(target_list, predict_list)
    recall = recall_score(target_list, predict_list)
    mcc = matthews_corrcoef(np.array([1 if x == 1 else -1 for x in target_list]),
                            np.array([1 if x == 1 else -1 for x in predict_list]))
    kappa = cohen_kappa_score(target_list, predict_list)
    auc = roc_auc_score(target_list, logit_list[:, 1])

    metric_meter.precision = precision
    metric_meter.recall = recall
    metric_meter.f1 = f1
    metric_meter.mcc = mcc
    metric_meter.kappa = kappa
    metric_meter.auc = auc

    return confusion


def save_checkpoint(model, path):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))


def get_optimizer(model, train_config, max_steps, warmup_steps):
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters()],
         'weight_decay': train_config['weight_decay'], 'lr': train_config['bert_lr']}, ], eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
    return optimizer, scheduler


if __name__ == '__main__':
    pass