import os
import torch
import pandas
import warnings
from torch_geometric.data import TemporalData

class TemporalPascalVOC:
    def __init__(self, root, name) -> None:
        self.root = root
        self.name = name.lower()
        # temporal_pascalvoc-sp_1024_10
        compactness = name.split("_")[-1]

        self.data_path = os.path.join(root, self.name, 'processed', f'{self.name}.pt')
        if os.path.exists(self.data_path):
            (self.train_data, self.val_data, self.test_data, 
             self.msg, self.t, self.num_nodes, self.num_events) = torch.load(self.data_path)
        else:
            os.makedirs(os.path.join(root, self.name, 'processed'), exist_ok=True)
            self.generate_data(compactness=compactness)
            tmp = (self.train_data, self.val_data, self.test_data, 
                   self.msg, self.t, self.num_nodes, self.num_events)
            torch.save(tmp, self.data_path)

    def generate_data(self, compactness):
        print("Generating data...")

        df = pandas.read_csv(os.path.join(self.root, f'voc_pascal_temporal_{compactness}.csv'))
        node_ft = pandas.read_csv(os.path.join(self.root, f'voc_pascal_temporal_nodefeatures_{compactness}.csv'))

        train, val, test = df[df.ext_roll==0], df[df.ext_roll==1], df[df.ext_roll==2]
        # node_ft = node_ft[node_ft.columns.drop(['x', 'y'])].values
        node_ft = node_ft.values
        x = torch.tensor(node_ft, dtype=torch.float32)

        tmp = [None, None, None]
        for i, data in enumerate([train, val, test]):
            t = torch.tensor(data.t.values, dtype=torch.long)
            tmp[i] = TemporalData(
                src = torch.tensor(data.src.values, dtype=torch.long),
                dst = torch.tensor(data.dst.values, dtype=torch.long),
                t = t,
                x = x,
                msg = torch.ones(size=(t.shape[0], 1), dtype=torch.float32),
                y = torch.tensor(data.label.values)
            )

        self.train_data, self.val_data, self.test_data = tmp

        self.t = torch.tensor(df.t.values, dtype=torch.long)
        self.msg = torch.ones(size=(self.t.shape[0], 1), dtype=torch.float32)
        self.num_nodes = x.shape[0]
        self.num_events = self.t.shape[0]
            
    def train_val_test_split(self, val_ratio=None, test_ratio=None):
        if val_ratio is not None or test_ratio is not None:
            warnings.warn(f'val_ratio and test_ratio are not used with {self.name} task.')

        return self.train_data, self.val_data, self.test_data
    