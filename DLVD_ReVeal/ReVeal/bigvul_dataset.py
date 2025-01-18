import copy
import os
import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import random
from torch.utils.data import Dataset


class BigVulGraphDataset(Dataset):
    """
    Due to the large size of the big-vul dataset,
    we employ a cache reading approach to prevent out-of-memory (OOM) errors
    when working with this dataset.
    """

    def __init__(self, data_path, mode='test'):

        self.mode = mode
        self.data_path = data_path
        self.cache_num = 15000
        self.cache = {}
        self.data, self.num_classes = self._parse_data(data_path)

    def __getitem__(self, index):

        data = self.data.iloc[index]
        if data['path'] in self.cache.keys():
            geom_data = self.cache[data['path']]
        else:
            with open(osp.join(self.data_path, data['path']), "rb") as f:
                gdata = pd.read_pickle(f)
                geom_data = gdata['geom_data'][0]
                if len(self.cache.keys()) < self.cache_num:
                    self.cache.update({f'{data["path"]}': geom_data})
                else:
                    random_key = random.choice(list(self.cache.keys()))
                    self.cache.pop(random_key)
                    self.cache.update({f'{data["path"]}': geom_data})
        return {
            "input_ids": geom_data.x.squeeze(0),
            "label": geom_data.y.squeeze(0),
            "edge_index": geom_data.edge_index.squeeze(0),
            "edge_type": geom_data.edge_attr.squeeze(0),
            "pseudo_label": torch.tensor(data['pseudo_label'], dtype=torch.long),
            "variance": torch.tensor(data['variance']),
            "index": index
        }

    def __len__(self):

        return len(self.data)

    def subseting_dataset(self, indices):

        self.old_data = copy.deepcopy(self.data)
        new_data = []
        for idx in indices:
            new_data.append(self.data[idx])

        self.data = new_data

        return self

    def update_data(self, index, content):
        for key in content.keys():
            self.data[key][index] = content[key]

    def _parse_data(self, data_path):
        data_path_list = os.listdir(data_path)
        total_num = len(data_path_list)
        dataset = {'path': data_path_list, 'index': range(0, len(data_path_list)),
                   'pseudo_label': [0] * len(data_path_list), 'variance': [0.0] * len(data_path_list)}
        dataset = pd.DataFrame(dataset)
        num_classes = 0
        label_set = set()
        for i, row in enumerate(dataset.itertuples()):
            with open(osp.join(data_path, row.path), "rb") as f:
                gdata = pd.read_pickle(f)
                geom_data = gdata['geom_data'][0]
                if len(self.cache.keys()) < self.cache_num:
                    self.cache.update({f'{row.path}': geom_data})
            X, label = geom_data.x, geom_data.y.item()
            label_set.add(label)
        num_classes = len(label_set)
        print(f'dataset num: {total_num}')
        return dataset, num_classes


def collate_fn(batch):
    label = []
    pseudo_label = []
    variance = []
    index = []
    pre_num = 0
    for i, data in enumerate(batch):
        label.append(data['label'])
        pseudo_label.append(data['pseudo_label'])
        variance.append(data['variance'])
        index.append(data['index'])
        if i == 0:
            input_ids = data['input_ids']
            edge_index = data['edge_index']
            edge_type = data['edge_type']
            continue
        pre_num += 400
        edge_index_tmp = data['edge_index'] + pre_num
        input_ids = torch.cat([input_ids, data['input_ids']], dim=0)
        edge_index = torch.cat([edge_index, edge_index_tmp], dim=1)
        edge_type = torch.cat([edge_type, data['edge_type']], dim=0)
    return {
        "input_ids": input_ids,
        "label": torch.tensor(label),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "pseudo_label": torch.tensor(pseudo_label),
        "variance": torch.tensor(variance),
        "index": index
    }


if __name__ == '__main__':
    if __name__ == '__main__':
        dataset = BigVulGraphDataset('VD_data/big-vul/0.1/train', mode='train')
        print(len(dataset))
        loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4)
        for i, data in enumerate(loader):
            print(data['input_ids'][0])
            print(data['edge_index'])
            print(data)
            if i == 4:
                break
