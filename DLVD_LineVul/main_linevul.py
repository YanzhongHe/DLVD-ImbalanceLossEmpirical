import argparse
import random
import os.path
import time
import datetime
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, \
    matthews_corrcoef, roc_auc_score
import torch.nn.functional as F
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, matthews_corrcoef, \
    cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaConfig, \
    RobertaForSequenceClassification

from TextDataset import TextDataset
from linevul.linevul_model import Model
from logger import TrainingLogger

from unbalanced_loss.LDAMLoss import LDAMLoss
from unbalanced_loss.focal_loss import BinaryFocalLoss
from unbalanced_loss.weight_ce_loss import WBCEWithLogitLoss
from unbalanced_loss.LALoss import LALoss
from unbalanced_loss.class_balanced_loss import Loss
from unbalanced_loss.DSCLoss import DSC_loss
from unbalanced_loss.DiceLoss import dice_loss
from unbalanced_loss.GHMC import GHMC


# from torch.utils.tensorboard import SummaryWriter


def train(train_dataset, model, args, val_dataset):
    file_name = f"{args.loss_comment}_{args.loss_comment}_{args.Sampling_comment}_events.out.tfevents"

    # tensorboard_log_path = os.path.join(args.data_path, args.comment, 'tf-logs', 'train')

    # output_file_path = os.path.join(tensorboard_log_path, file_name)

    # writer = SummaryWriter(output_file_path)

    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epoch * len(train_dataloader)

    # evaluate the model per epoch
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Print arguments
    for k, v in sorted(vars(args).items()):
        logger.print(f'{str(k)}={str(v)}')

    # Train
    logger.print("***** Running training *****")
    logger.print("  Num examples = %d" % len(train_dataset))
    logger.print("  Num Epochs = %d" % args.epoch)
    logger.print("  Instantaneous batch size per GPU = %d" % (args.train_batch_size // max(args.n_gpu, 1)))
    logger.print("  Total train batch size = %d" % args.train_batch_size * args.gradient_accumulation_steps)
    logger.print("  Gradient Accumulation steps = %d" % args.gradient_accumulation_steps)
    logger.print("  Total optimization steps = %d" % args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epoch):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (input_ids, labels) = [x.to(args.device) for x in batch]
            model.train()

            loss, logits = model(input_ids=input_ids, labels=labels)
            # loss = args.loss_fct(logits, labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # writer.add_scalar('Train/Loss', loss.item(), global_step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description(f"epoch {idx} loss {avg_loss}")

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, val_dataset, eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.print("  " + "*" * 20)
                        logger.print("  Best f1:%s" % round(best_f1, 4))
                        logger.print("  " + "*" * 20)

                        desired_part = args.data_path.split('data/')[1]

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = f'{args.save_path}/{args.loss_comment}/{checkpoint_prefix}/{desired_part}'
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, args.loss_comment)
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.print("Saving model checkpoint to %s" % output_dir)

    # writer.close()


def evaluate(args, model, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    val_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.print("***** Running evaluation *****")
    logger.print("  Num examples = %d" % len(val_dataloader))
    logger.print("  Batch size = %d" % args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in val_dataloader:
        (inputs_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            # lm_loss = args.loss_fct(logits, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits[:, 1] > best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    mcc = matthews_corrcoef(y_trues, y_preds)
    kappa = cohen_kappa_score(y_trues, y_preds)
    g_measure = np.sqrt(precision * recall)
    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "g_measure": float(g_measure),
        "eval_threshold": best_threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    logger.print("***** Eval results *****")
    for key in sorted(result.keys()):
        log_message = "  {} = {:.4f}".format(key, result[key])
        logger.print(log_message)

    return result


def test(args, model, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.print("***** Running Test *****")
    logger.print(f"  Num examples = {len(test_dataset)}")
    logger.print(f"  Batch size = {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []

    results_df = pd.DataFrame(columns=["True_Label", "Predicted_Label"])

    for batch in test_dataloader:
        (inputs_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            # print(len(inputs_ids))
            # print(len(labels))
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            # lm_loss = args.loss_fct(logits, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

            batch_df = pd.DataFrame({
                "True_Label": labels.cpu().numpy(),
                "Predicted_Label": (logit.cpu().numpy()[:, 1] > best_threshold).astype(int)
            })
            results_df = pd.concat([results_df, batch_df], ignore_index=True)

        nb_eval_steps += 1

    results_df.to_csv(
        f"{args.data_path}/{args.loss_comment}/{args.loss_comment}_{args.Sampling_comment}_test_results.csv",
        index=False)

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    mcc = matthews_corrcoef(y_trues, y_preds)
    kappa = cohen_kappa_score(y_trues, y_preds)
    g_measure = np.sqrt(precision * recall)
    auc = roc_auc_score(y_trues, logits[:, 1])
    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "g_measure": float(g_measure),
        "test_threshold": best_threshold,
        "test_auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    logger.print("***** Test results *****")
    for key in sorted(result.keys()):
        log_message = "  {} = {:.4f}".format(key, result[key])
        logger.print(log_message)


def set_random_seed(seed):
    # print('Random seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


data_paths = ['./data/Devign/1.0','./data/reveal/1.0','./data/big-vul/1.0','./data/Juliet/1.0']

seeds = [1, 2, 3, 4, 5]
loss_functions = ['celoss','LAloss','LDAMloss','WCEloss','GHMloss','focalloss','CBceloss','CBfocalloss','DiceLoss','DSCLoss']


for loss_fct_1 in loss_functions:
    for data_path in data_paths:
        for seed in seeds:
            if __name__ == "__main__":

                set_random_seed(42)

                parser = argparse.ArgumentParser()
                parser.add_argument("--train_batch_size", default=16, type=int, required=False,
                                    help="Batch size per GPU/CPU for training.")
                parser.add_argument("--eval_batch_size", default=16, type=int, required=False,
                                    help="Batch size per GPU/CPU for evaluation.")
                parser.add_argument("--learning_rate", default=2e-5, type=float, required=False,
                                    help="The initial learning rate for Adam.")
                parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=False,
                                    help="Epsilon for Adam optimizer.")
                parser.add_argument("--save_path", default='./saved_models', type=str, required=False,
                                    help="output dir")
                parser.add_argument("--data_path", default='./data/Devign', type=str, required=False,
                                    help="data path")
                parser.add_argument("--loss_comment", default='celoss', type=str, required=False,
                                    help="loss fct")
                parser.add_argument("--Sampling_comment", default='None', type=str, required=False,
                                    help="Sampling")
                parser.add_argument("--tokenizer_path", default='./bert', type=str, required=False,
                                    help="tokenizer path")
                parser.add_argument("--block_size", default=512, type=int, required=False,
                                    help="")
                parser.add_argument('--epoch', type=int, default=10,
                                    help="training epochs")
                parser.add_argument('--gradient_accumulation_steps', type=int, default=1, required=False,
                                    help="Number of updates steps to accumulate before performing a backward/update pass.")
                parser.add_argument('--comment', type=str, default='default', required=False,
                                    help="Description for current work")
                parser.add_argument('--model_name_or_path', type=str, default='./bert', required=False,
                                    help="")
                parser.add_argument('--model_name', type=str, default='12heads_linevul_model.bin', required=False,
                                    help="")
                parser.add_argument("--weight_decay", default=0.0, type=float,
                                    help="Weight deay if we apply some.")
                parser.add_argument("--max_grad_norm", default=1.0, type=float,
                                    help="Max gradient norm.")
                args = parser.parse_args()

                seed_directory = f'seed{seed}'
                args.data_path = os.path.join(data_path, seed_directory)

                args.loss_comment = loss_fct_1

                # logger
                if not os.path.exists(f'{args.data_path}/{args.loss_comment}'):
                    os.makedirs(f'{args.data_path}/{args.loss_comment}')
                logger = TrainingLogger(f'{args.data_path}/{args.loss_comment}/{datetime.date.today()}')
                args.logger = logger

                logger.print("---------TIME:%s----------" % str(datetime.datetime.now()))
                logger.print(
                    "---------Settings-> data_path: %s, loss_comment: %s, Sampling_comment: %s, comment: %s ----------" % (
                        args.data_path, args.loss_comment, args.Sampling_comment, args.loss_comment))

                # cuda
                USE_CUDA = torch.cuda.is_available()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                args.n_gpu = torch.cuda.device_count()
                args.device = device

                # tokenizer
                tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)



                # linevul
                config = RobertaConfig.from_pretrained(args.model_name_or_path)
                config.num_labels = 1
                config.num_attention_heads = 12
                model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config,
                                                                         ignore_mismatched_sizes=True)

                train_dataset = TextDataset(tokenizer, args, 'train')
                val_dataset = TextDataset(tokenizer, args, 'val')
                test_dataset = TextDataset(tokenizer, args, 'test')

                samples_per_class = train_dataset.samples_per_class
                print(samples_per_class)

                negative_samples = samples_per_class[0]
                positive_samples = samples_per_class[1]

                total_samples = sum(samples_per_class)

                negative_ratio = negative_samples / total_samples
                positive_ratio = positive_samples / total_samples

                args.negative_ratio = negative_ratio
                args.positive_ratio = positive_ratio

                # loss function
                if args.loss_comment == 'focalloss':

                    print("###########***")
                    print(args.negative_ratio)

                    args.loss_fct = Loss(
                        loss_type="focal_loss_0",
                        fl_gamma=2,  # focal loss gamma
                        class_balanced=False
                    )


                elif args.loss_comment == 'celoss':
                    args.loss_fct = nn.CrossEntropyLoss()
                    print("1")


                elif args.loss_comment == 'CBceloss':
                    args.loss_fct = Loss(
                        loss_type="binary_cross_entropy",
                        beta=0.999,  # class-balanced loss beta
                        fl_gamma=2,  # focal loss gamma
                        samples_per_class=samples_per_class,
                        class_balanced=True
                    )


                elif args.loss_comment == 'CBfocalloss':
                    args.loss_fct = Loss(
                        loss_type="focal_loss",
                        beta=0.999,  # class-balanced loss beta
                        fl_gamma=2,  # focal loss gamma
                        samples_per_class=samples_per_class,
                        class_balanced=True
                    )

                elif args.loss_comment == 'GHMloss':
                    args.loss_fct = GHMC(bins=10, momentum=0.9, use_sigmoid=True, loss_weight=1.0)

                elif args.loss_comment == 'DSCLoss':
                    args.loss_fct = DSC_loss

                elif args.loss_comment == 'DiceLoss':
                    args.loss_fct = dice_loss

                elif args.loss_comment == 'LDAMloss':
                    args.loss_fct = LDAMLoss(cls_num_list=samples_per_class)

                elif args.loss_comment == 'LAloss':
                    cls_num_list = np.array(samples_per_class)
                    cls_num_list = torch.from_numpy(cls_num_list)
                    args.loss_fct = LALoss(cls_num_list=cls_num_list)

                elif args.loss_comment == 'WCEloss':


                    negative_weight = 1.0 / negative_samples
                    positive_weight = 1.0 / positive_samples

                    class_weights = torch.tensor([negative_weight, positive_weight], dtype=torch.float).to(device)

                    args.loss_fct = nn.CrossEntropyLoss(weight=class_weights)

                model = Model(model, config, tokenizer, args)

                # train
                train(train_dataset, model, args, val_dataset)

                # test
                results = {}
                checkpoint_prefix = f'checkpoint-best-f1'
                desired_part = args.data_path.split('data/')[1]
                output_dir = f'{args.save_path}/{args.loss_comment}/{checkpoint_prefix}/{desired_part}/{args.loss_comment}'
                print("#############")
                print(output_dir)
                model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
                model.to(args.device)
                test(args, model, test_dataset, 0.5)

