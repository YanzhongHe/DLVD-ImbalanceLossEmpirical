import json
import os.path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


def convert_examples_to_features(func, label, tokenizer, args):
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        # if file_type == "train":
        #     file_path = os.path.join(args.data_path, f'{file_type}.json')
        # elif file_type == "eval":
        #     file_path = args.eval_data_file
        # elif file_type == "test":
        #     file_path = args.test_data_file
        self.examples = []
        self.samples_per_class = [0, 0]  # Initialize with two classes [0, 1]
        df = pd.read_json(os.path.join(args.data_path, f'{file_type}.json'))

        funcs = df['processed_func'].tolist()
        labels = df['target'].tolist()

        for i in tqdm(range(len(funcs))):
            label = labels[i]
            self.samples_per_class[label] += 1  # Increment count for the current class
            self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

