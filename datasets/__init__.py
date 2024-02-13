import torch

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from .label_prediction_sequence import LabelSequencePrediction
from torch_geometric.datasets import JODIEDataset
from torch_geometric.data import TemporalData
from .pascalvoc import TemporalPascalVOC
from numpy.random import default_rng

import numpy


PASCALVOC = ["Temporal_PascalVOC-SP_1024_10", "Temporal_PascalVOC-SP_1024_30"]
JODIE = ['Wikipedia', "Reddit", "MOOC", "LastFM"]

SYNTHETIC_SEQUENCE = [
    "LabelSequence_3_1000", "LabelSequence_5_1000", "LabelSequence_7_1000", 
    "LabelSequence_9_1000", "LabelSequence_11_1000", "LabelSequence_15_1000",
    "LabelSequence_20_1000"
]

TGB_DATASETS = [
    "tgbl-wiki",
    "tgbl-review",
    "tgbl-coin", 
    "tgbl-comment" 
] 

DATA_NAMES = JODIE + SYNTHETIC_SEQUENCE + PASCALVOC + TGB_DATASETS

def get_dataset(root, name, seed):
    rng = default_rng(seed)

    dataset = None

    if name in JODIE:
        dataset = JODIEDataset(root, name)
        data = dataset[0]
        data.x = torch.tensor(rng.random((data.num_nodes, 1), dtype=numpy.float32))
        num_nodes, edge_dim = data.num_nodes, data.msg.shape[-1] 
        node_dim = data.x.shape[-1] #if hasattr(data, 'x') else 0
        init_time = data.t[0]
        out_dim = 1

    elif name in SYNTHETIC_SEQUENCE :
        (data, num_nodes, edge_dim, node_dim,
            out_dim, init_time) = synthetic(root, name, seed)
         
    elif name in PASCALVOC:
        data = TemporalPascalVOC(root=root, name=name)
        num_nodes = data.num_nodes
        edge_dim, node_dim = data.train_data.msg.shape[-1], data.train_data.x.shape[-1]
        init_time = data.train_data.t[0]
        out_dim = max(data.train_data.y.max(), data.val_data.y.max(), data.test_data.y.max()) + 1 
        assert out_dim == 21, "The number of classes is exactly 21 for this dataset, this should not happen"

    elif name in TGB_DATASETS:
        dataset = PyGLinkPropPredDataset(root=root, name=name, absolute_path=True)
        data: TemporalData = dataset.get_TemporalData()
        num_nodes = data.num_nodes
        edge_dim = data.msg.shape[-1]
        node_dim = 1
        out_dim = 1
        init_time = data.t[0]

    else:
        raise NotImplementedError
    
    return data, num_nodes, edge_dim, node_dim, out_dim, init_time, dataset


def synthetic(root, name, seed):
    spl = name.split('_')
    num_seq = int(spl[-2])
    seq_len = int(spl[-3])
    num_feats = int(spl[-1])
 
    if 'LabelSequence' in name: 
        data = LabelSequencePrediction(root=root, name=name, seq_len=seq_len, num_seq=num_seq, feat_dim=num_feats, seed=seed)
        num_nodes = data.num_nodes
        edge_dim, node_dim = data.data[0].msg.shape[-1], data.data[0].x.shape[-1]
        init_time = data.data[0].t[0]
        out_dim = data.data[0].y.shape[-1]
        assert out_dim == 1
    else:
        raise ValueError(f'The name is not in {SYNTHETIC_SEQUENCE}. Got {name}')
    return data, num_nodes, edge_dim, node_dim, out_dim, init_time
