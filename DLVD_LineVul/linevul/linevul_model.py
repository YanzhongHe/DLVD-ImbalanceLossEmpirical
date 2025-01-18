import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification
import unbalanced_loss.LDAMLoss
from unbalanced_loss.focal_loss import BinaryFocalLoss
from unbalanced_loss.weight_ce_loss import WBCEWithLogitLoss
from unbalanced_loss.dice_loss_nlp import BinaryDSCLoss
import unbalanced_loss.LALoss
from unbalanced_loss.GHM_loss import GHMC_Loss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(RobertaForSequenceClassification):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        print("################")
        print("args.loss_fct:", args.loss_fct)

    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):

        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1),
                                               output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                if self.args.loss_comment == "GHMloss":

                    neg_weight = 1.0 / self.args.negative_ratio
                    pos_weight = 1.0 / self.args.positive_ratio

                    batch_size = logits.size(0)
                    label_weight = torch.zeros((batch_size, 2), device=logits.device)  # 2表示两个类别

                    label_weight[:, 0] = neg_weight
                    label_weight[:, 1] = pos_weight

                    class_num = 2

                    expanded_labels = torch.zeros((labels.size(0), class_num), device=labels.device)
                    expanded_labels[range(len(labels)), labels] = 1

                    loss_fct = self.args.loss_fct
                    loss = loss_fct(logits, expanded_labels, label_weight)

                elif self.args.loss_comment == "focalloss":

                    neg_weight = 1.0 / self.args.negative_ratio
                    pos_weight = 1.0 / self.args.positive_ratio

                    batch_size = logits.size(0)
                    label_weight = torch.zeros((batch_size, 2), device=logits.device)  # 2表示两个类别

                    label_weight[:, 0] = neg_weight
                    label_weight[:, 1] = pos_weight

                    class_num = 2

                    loss_fct = self.args.loss_fct
                    loss = loss_fct(logits, labels, label_weight)

                else:
                    loss_fct = self.args.loss_fct
                    loss = loss_fct(logits, labels)

                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = \
                    self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1),
                                         output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                if self.args.loss_comment == "GHMloss":

                    neg_weight = 1.0 / self.args.negative_ratio
                    pos_weight = 1.0 / self.args.positive_ratio

                    batch_size = logits.size(0)
                    label_weight = torch.zeros((batch_size, 2), device=logits.device)  # 2表示两个类别

                    label_weight[:, 0] = neg_weight
                    label_weight[:, 1] = pos_weight

                    class_num = 2

                    expanded_labels = torch.zeros((labels.size(0), class_num), device=labels.device)
                    expanded_labels[range(len(labels)), labels] = 1

                    loss_fct = self.args.loss_fct
                    loss = loss_fct(logits, expanded_labels, label_weight)

                elif self.args.loss_comment == "focalloss":

                    neg_weight = 1.0 / self.args.negative_ratio
                    pos_weight = 1.0 / self.args.positive_ratio

                    batch_size = logits.size(0)
                    label_weight = torch.zeros((batch_size, 2), device=logits.device)

                    label_weight[:, 0] = neg_weight
                    label_weight[:, 1] = pos_weight

                    class_num = 2

                    loss_fct = self.args.loss_fct
                    loss = loss_fct(logits, labels, label_weight)
                else:
                    loss_fct = self.args.loss_fct
                    loss = loss_fct(logits, labels)

                return loss, prob
            else:
                return prob
