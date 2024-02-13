import os
import torch
import numpy
from numpy.random import default_rng
from torch_geometric.data import TemporalData


class LabelSequencePrediction:
    def __init__(self, root, name, seq_len, num_seq=1000, seed=9, feat_dim=1) -> None:
        self.rng = default_rng(seed)
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.feat_dim = feat_dim

        assert self.seq_len and self.num_seq

        self.data_path = os.path.join(root, name.lower(), 'processed', f'labelprediction_seq_len{seq_len}_num_seq{num_seq}_featdim{feat_dim}_seed{seed}.pt')
        if os.path.exists(self.data_path):
            self.data = torch.load(self.data_path)
        else:
            os.makedirs(os.path.join(root, name.lower(), 'processed'), exist_ok=True)
            self.generate_data()
            torch.save(self.data, self.data_path)
        self.num_nodes = (self.seq_len + 1) * self.num_seq
        self.num_events = self.num_seq * self.num_seq

    def generate_data(self):
        print("Generating data...")

        self.data = []
        first_src_id, last_src_id = 0, self.seq_len - 1
        
        # Generate random input node features 
        x = self.rng.uniform(size=(self.seq_len * self.num_seq, self.feat_dim), low=-1., high=1.)
        # x = numpy.zeros(self.seq_len * self.num_seq)
        signals = torch.tensor([(i % 2)*2-1 for i in range(self.num_seq)], dtype=torch.float32)  # (num_seq,)
        signals = signals.reshape((-1, 1)).tile((1, self.feat_dim))  # (num_seq, feat_dim)
        x[::self.seq_len] = signals
        x = x.reshape(1, self.seq_len * self.num_seq, -1) # in each batch we want the possibility to look at all the node initial features
        x = torch.from_numpy(x).type(torch.float32)

        for i in range(self.num_seq):
            # Generate random input edge features 
            msg = self.rng.uniform(size=(self.seq_len-1, self.feat_dim), low=-1., high=1.)
            msg[0] = (i % 2)*2-1  # same signal as for x
            msg = torch.tensor(msg, dtype=torch.float32)
            
            d = TemporalData(
                src = torch.arange(first_src_id, last_src_id, dtype=torch.long),
                dst = torch.arange(first_src_id + 1, last_src_id + 1, dtype=torch.long),
                t = torch.arange(msg.shape[0], dtype=torch.long),
                x = x,
                msg = msg,
                y = torch.tensor([i % 2], dtype=torch.float32)
            )
            self.data.append(d)
            first_src_id, last_src_id = last_src_id + 1, self.seq_len + last_src_id
        
    def train_val_test_split(self, val_ratio, test_ratio):
        train_size = int(len(self.data) * (1 - val_ratio - test_ratio))
        val_size = int(len(self.data) * val_ratio) + train_size
        
        train_data = self.data[:train_size]
        val_data = self.data[train_size: val_size]
        test_data = self.data[val_size:]

        return train_data, val_data, test_data
    
