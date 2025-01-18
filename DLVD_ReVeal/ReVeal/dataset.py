import copy
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from torch.utils.data import Dataset


class GraphDataset(Dataset):

    def __init__(self, data_path, mode='test'):

        self.mode = mode
        self.data_path = data_path
        self.data, self.num_classes = self._parse_data(data_path)

    def __getitem__(self, index):

        data = self.data.iloc[index]
        geom_data = data['geom_data']

        return {
            "input_ids": geom_data.x.squeeze(0),
            "label": geom_data.y.squeeze(0),
            "edge_index": geom_data.edge_index.squeeze(0),
            "edge_type": geom_data.edge_attr.squeeze(0),
            "pseudo_label": torch.tensor(data['pseudo_label'], dtype=torch.long),
            "variance": torch.tensor(data['variance']),
            "index": data['index']
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
        total_num = 0
        with open(data_path, "rb") as f:
            dataset = pd.read_pickle(f)
            dataset = dataset[['geom_data']]
            total_num += len(dataset)
            dataset['pseudo_label'] = 0
            dataset['variance'] = 0.0

            dataset.reset_index(inplace=True)
            dataset['index'] = dataset.index
        num_classes = 0
        if self.mode == 'train':
            label_set = set()
            for i, row in enumerate(dataset.itertuples()):
                geom_data = row.geom_data
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
    pass
