import torch

from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TimeEncoder
from torch_geometric.nn.resolver import activation_resolver
from typing import Callable, Optional, Any, Dict, Union, List
from .predictors import *
from .memory_layers import *
from .gnn_layers import *
import numpy as np


class GenericModel(torch.nn.Module):
    
    def __init__(self, num_nodes, memory=None, gnn=None, gnn_act=None, readout=None, predict_dst=False):
        super(GenericModel, self).__init__()
        self.memory = memory
        self.gnn = gnn
        self.gnn_act = gnn_act
        self.readout = readout
        self.num_gnn_layers = 1
        self.num_nodes = num_nodes
        self.predict_dst = predict_dst

    def reset_memory(self):
        if self.memory is not None: self.memory.reset_state()

    def zero_grad_memory(self):
        if self.memory is not None: self.memory.zero_grad_memory()

    def update(self, src, pos_dst, t, msg, *args, **kwargs):
        if self.memory is not None: self.memory.update_state(src, pos_dst, t, msg)

    def detach_memory(self):
        if self.memory is not None: self.memory.detach()
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        super().reset_parameters()
        if hasattr(self.memory, 'reset_parameters'):
            self.memory.reset_parameters()
        if hasattr(self.gnn, 'reset_parameters'):
                    self.gnn.reset_parameters()
        if hasattr(self.readout, 'reset_parameters'):
                    self.readout.reset_parameters()

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        # Get updated memory of all nodes involved in the computation.
        m, last_update = self.memory(n_id)

        if hasattr(batch, 'x'):
            if len(batch.x.shape) == 3: # sequence classification case
                x = batch.x.squeeze(0)
            elif len(batch.x.shape) == 2: # link-based predictions
                x = batch.x
            else:
                raise ValueError(f"Unexpected node feature shape. Got {batch.x.shape}")
            z = torch.cat((m, x[n_id]), dim=-1)

        if self.gnn is not None:
            for gnn_layer in self.gnn:
                z = gnn_layer(z, last_update, edge_index, t, msg)
                z = self.gnn_act(z)

        if self.predict_dst:
            pos_out = self.readout(z[id_mapper[pos_dst]])
            neg_dst = None
        else:
            pos_out = self.readout(z[id_mapper[src]], z[id_mapper[pos_dst]])
            neg_out = self.readout(z[id_mapper[src]], z[id_mapper[neg_dst]]) if neg_dst is not None else None

        return pos_out, neg_out, m[id_mapper[src]], m[id_mapper[pos_dst]]
    

class TGN(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int, 
                 edge_dim: int, 
                 memory_dim: int, 
                 time_dim: int, 
                 node_dim: int = 0, 
                 # GNN params
                 gnn_hidden_dim: List[int] = [],
                 gnn_act: Union[str, Callable, None] = 'tanh',
                 gnn_act_kwargs: Optional[Dict[str, Any]] = None,
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 readout_out: Optional[int] = 1,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0,
                 # Distinguish between link prediction and sequence prediction
                 predict_dst: bool = False,
                 # conv type Transformer or Simple conv
                 conv_type: str = 'TransformerConv'
        ):

        assert conv_type in ['TransformerConv']

        # Define memory
        memory = GeneralMemory(
            num_nodes,
            edge_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(edge_dim, memory_dim, time_dim), # TODO we can change this with a MLP
            aggregator_module=LastAggregator(),
            rnn='GRUCell',
            init_time=init_time
        )

        # Define GNN
        gnn = torch.nn.Sequential()
        gnn_act = activation_resolver(gnn_act, **(gnn_act_kwargs or {}))
        h_prev = memory_dim + node_dim
        for h in gnn_hidden_dim:
            gnn.append(GraphAttentionEmbedding(h_prev, h, edge_dim, time_enc=memory.time_enc, 
                                               mean_delta_t=mean_delta_t, std_delta_t=std_delta_t, conv_type=conv_type))
            h_prev = h * 2 # We double the input dimension because GraphAttentionEmbedding has 2 concatenated heads

        # Define the link predictor
        # NOTE: We double the input dimension because GraphAttentionEmbedding with TransformerConv has 2 concatenated heads
        in_channels = gnn_hidden_dim[-1] * 2
        readout = (LinkPredictor(in_channels, readout_hidden, readout_out) if not predict_dst 
                   else SequencePredictor(in_channels, readout_hidden, readout_out))

        super().__init__(num_nodes, memory, gnn, gnn_act, readout, predict_dst)
        self.num_gnn_layers = len(gnn_hidden_dim)


class DyRep(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int,
                 edge_dim: int, 
                 memory_dim: int, 
                 node_dim: int = 0, 
                 non_linearity: str = 'tanh',
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 readout_out: Optional[int] = 1,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0,
                 # Distinguish between link prediction and sequence prediction
                 predict_dst: bool = False,
                 # conv type Transformer
                 conv_type: str = 'TransformerConv'
        ):

        assert conv_type in ['TransformerConv']

        # Define memory
        memory = DyRepMemory(
            num_nodes,
            edge_dim,
            memory_dim,
            message_module=DyRepMessage(edge_dim, memory_dim, 1),
            aggregator_module=LastAggregator(),
            non_linearity=non_linearity,
            mean_delta_t=mean_delta_t, 
            std_delta_t=std_delta_t,
            init_time=init_time, 
            conv_type = conv_type
        )
       
        # Define the link predictor
        readout = (LinkPredictor(memory_dim + node_dim, readout_hidden, readout_out) if not predict_dst 
                   else SequencePredictor(memory_dim + node_dim, readout_hidden, readout_out))

        super().__init__(num_nodes, memory, readout=readout, predict_dst=predict_dst)
        self.num_gnn_layers = 1


class JODIE(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int,
                 edge_dim: int, 
                 memory_dim: int,
                 time_dim: int,
                 node_dim: int = 0, 
                 non_linearity: str = 'tanh',
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 readout_out: Optional[int] = 1,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0,
                 # Distinguish between link prediction and sequence prediction
                 predict_dst: bool = False
        ):
        # Define memory
        memory = GeneralMemory(
            num_nodes,
            edge_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(edge_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
            rnn='RNNCell',
            non_linearity=non_linearity,
            init_time = init_time
        )

        # Define the link predictor
        readout = (LinkPredictor(memory_dim + node_dim, readout_hidden, readout_out) if not predict_dst 
                   else SequencePredictor(memory_dim + node_dim, readout_hidden, readout_out))

        super().__init__(num_nodes, memory, readout=readout, predict_dst=predict_dst)
        self.num_gnn_layers = 1
        self.projector_src = JodieEmbedding(memory_dim + node_dim, mean_delta_t=mean_delta_t, std_delta_t=std_delta_t)
        self.projector_dst = JodieEmbedding(memory_dim + node_dim, mean_delta_t=mean_delta_t, std_delta_t=std_delta_t)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self.projector_src, 'reset_parameters'):
            self.projector_src.reset_parameters()
        if hasattr(self.projector_dst, 'reset_parameters'):
                    self.projector_dst.reset_parameters()
       

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        # Get updated memory of all nodes involved in the computation.
        m, last_update = self.memory(n_id)

        if hasattr(batch, 'x'):
            if len(batch.x.shape) == 3: # sequence classification case
                x = batch.x.squeeze(0)
            elif len(batch.x.shape) == 2: # link-based predictions
                x = batch.x
            else:
                raise ValueError(f"Unexpected node feature shape. Got {batch.x.shape}")
            z = torch.cat((m, x[n_id]), dim=-1)

        # Compute the projected embeddings
        z_src =  self.projector_src(z[id_mapper[src]], last_update[id_mapper[src]], batch.t)
        z_pos_dst =  self.projector_dst(z[id_mapper[pos_dst]], last_update[id_mapper[pos_dst]], batch.t)

        if self.predict_dst:
            pos_out = self.readout(z_pos_dst)        
        else:
            pos_out = self.readout(z_src, z_pos_dst)

        if neg_dst is not None:
            z_neg_dst =  self.projector_dst(z[id_mapper[neg_dst]], last_update[id_mapper[neg_dst]], batch.t)
            neg_out = self.readout(z_src, z_neg_dst)
        else:
            neg_out = None

        return pos_out, neg_out, m[id_mapper[src]], m[id_mapper[pos_dst]]


class TGAT(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int, 
                 edge_dim: int, 
                 time_dim: int, 
                 node_dim: int = 0, 
                 # GNN params
                 gnn_hidden_dim: List[int] = [],
                 gnn_act: Union[str, Callable, None] = 'relu',
                 gnn_act_kwargs: Optional[Dict[str, Any]] = None,
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 readout_out: Optional[int] = 1,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0,
                 # Distinguish between link prediction and sequence prediction
                 predict_dst: bool = False,
                 # conv type Transformer
                 conv_type: str = 'TransformerConv'
        ):

        assert conv_type in ['TransformerConv']

        memory = LastUpdateMemory(num_nodes, init_time)

        # Define GNN
        gnn = torch.nn.Sequential()
        gnn_act = activation_resolver(gnn_act, **(gnn_act_kwargs or {}))
        h_prev = node_dim
        for h in gnn_hidden_dim:
            gnn.append(GraphAttentionEmbedding(h_prev, h, edge_dim, time_enc=TimeEncoder(time_dim), 
                                               mean_delta_t=mean_delta_t, std_delta_t=std_delta_t, conv_type=conv_type))
            h_prev = h * 2 # We double the input dimension because GraphAttentionEmbedding has 2 concatenated heads


        # Define the link predictor
        # NOTE: We double the input dimension because GraphAttentionEmbedding with TransformerConv has 2 concatenated heads
        in_channels = gnn_hidden_dim[-1] * 2
        readout = (LinkPredictor(in_channels, readout_hidden, readout_out) if not predict_dst 
                   else SequencePredictor(in_channels, readout_hidden, readout_out))

        super().__init__(num_nodes, memory, gnn, gnn_act, readout, predict_dst=predict_dst)
        self.num_gnn_layers = len(gnn_hidden_dim)

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        last_update = self.memory(n_id)

        if len(batch.x.shape) == 3: # sequence classification case
            x = batch.x.squeeze(0)
        elif len(batch.x.shape) == 2: # link-based predictions
            x = batch.x
        else:
            raise ValueError(f"Unexpected node feature shape. Got {batch.x.shape}")
        z = x[n_id]

        if self.gnn is not None:
            for gnn_layer in self.gnn:
                z = gnn_layer(z, last_update, edge_index, t, msg)
                z = self.gnn_act(z)

        if self.predict_dst:
            pos_out = self.readout(z[id_mapper[pos_dst]])
            neg_out = None   
        else:
            pos_out = self.readout(z[id_mapper[src]], z[id_mapper[pos_dst]])
            neg_out = self.readout(z[id_mapper[src]], z[id_mapper[neg_dst]]) if neg_dst is not None else None

        return pos_out, neg_out, None, None
    

class EdgeBank(GenericModel):
    def __init__(self, num_nodes, timespan, time_window: float = np.inf):
        # NOTE: 
        # when time_window == np.inf it uses the entire data, 
        # while if 0 < time_window <= 1 then edgebank uses only a percentage of the timespan
        
        super().__init__(num_nodes, memory=None, gnn=None, gnn_act=None, readout=None)
        self.edgebank = {}
        
        # NOTE: best way to compute the time window: 
        # time_window = int((train_timestamp.max() - train_timestamp.min()) * time_window_ratio)
        self.time_window = int(timespan * time_window)
        self.num_gnn_layers = torch.nn.parameter.Parameter(torch.tensor(0), requires_grad=False)
    
    def update(self, src, pos_dst, t, msg, src_emb, pos_dst_emb):
        for s, d, tt in zip(src, pos_dst, t):
            self.edgebank[(s.cpu().item(), d.cpu().item())] = tt.cpu().item() 

    def _predict(self, src, dst, t):
        prev_t = t - self.time_window
        prev_t[prev_t < 0] = 0
        out = torch.zeros(src.size())
        for i, (s, d, tt) in enumerate(zip(src, dst, prev_t)):
            if self.edgebank[(s.cpu().item(), d.cpu().item())] >= tt:
                out[i] = 1
            else:
                out[i] = 0
        return out.unsqueeze(1)

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        pos_out = self._predict(src, pos_dst, batch.t)
        neg_out = self._predict(src, neg_dst, batch.t) if neg_dst is not None else None
        emb_src, emb_pos_dst = None, None

        return pos_out, neg_out, emb_src, emb_pos_dst


class CTAN(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int, 
                 edge_dim: int, 
                 memory_dim: int, 
                 time_dim: int,
                 node_dim: int = 0, 
                 # CTAN params
                 num_iters: int = 1,
                 gnn_act: Union[str, Callable, None] = 'tanh',
                 gnn_act_kwargs: Optional[Dict[str, Any]] = None,
                 epsilon: float = 0.1,
                 gamma: float = 0.1,
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 readout_out: Optional[int] = 1,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0,
                 final_act: Optional[str] = None,
                 # Distinguish between link prediction and sequence prediction
                 predict_dst: bool = False,
                 # conv type Transformer
                 conv_type: str = 'TransformerConv'
        ):

        assert conv_type in ['TransformerConv']

        # Define memory
        memory = SimpleMemory(
            num_nodes,
            memory_dim,
            aggregator_module=LastAggregator(),
            init_time=init_time
        )

        gnn = CTANEmbedding(memory_dim, node_dim, edge_dim, time_dim, num_iters, gnn_act, gnn_act_kwargs, 
                            epsilon, gamma, mean_delta_t,  std_delta_t, conv_type=conv_type)
        
        # Define the link predictor
        readout = (LinkPredictor(memory_dim, readout_hidden, readout_out) if not predict_dst 
                   else SequencePredictor(memory_dim, readout_hidden, readout_out))

        super().__init__(num_nodes, memory, gnn, gnn_act, readout, predict_dst=predict_dst)
        self.num_gnn_layers = num_iters

        self.final_act = getattr(torch, final_act) if final_act is not None else None

    def zero_grad_memory(self):
        if self.memory is not None: self.memory.zero_grad_memory()
        
    def update(self, src, pos_dst, t, msg, src_emb, pos_dst_emb):
        self.memory.update_state(src, pos_dst, t, src_emb, pos_dst_emb)

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        # Get updated memory of all nodes involved in the computation.
        z, last_update = self.memory(n_id)

        if hasattr(batch, 'x'):
            if len(batch.x.shape) == 3: # sequence classification case
                x = batch.x.squeeze(0)
            elif len(batch.x.shape) == 2: # link-based predictions
                x = batch.x
            else:
                raise ValueError(f"Unexpected node feature shape. Got {batch.x.shape}")
            z = torch.cat((z, x[n_id]), dim=-1)

        z = self.gnn(z, last_update, edge_index, t, msg)

        if self.final_act is not None:
            z = self.final_act(z)
        
        if self.predict_dst:
            pos_out = self.readout(z[id_mapper[pos_dst]])
            neg_out = None   
        else:
            pos_out = self.readout(z[id_mapper[src]], z[id_mapper[pos_dst]])
            neg_out = self.readout(z[id_mapper[src]], z[id_mapper[neg_dst]]) if neg_dst is not None else None

        emb_src = z[id_mapper[src]]
        emb_pos_dst = z[id_mapper[pos_dst]]

        return pos_out, neg_out, emb_src, emb_pos_dst
    


class CTAN_tgb(CTAN):
    def __init__(self, 
                # Memory params
                 num_nodes: int, 
                 edge_dim: int, 
                 memory_dim: int, 
                 time_dim: int,
                 node_dim: int = 0, 
                 # CTAN params
                 num_iters: int = 1,
                 gnn_act: Union[str, Callable, None] = 'tanh',
                 gnn_act_kwargs: Optional[Dict[str, Any]] = None,
                 epsilon: float = 0.1,
                 gamma: float = 0.1,
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 readout_out: Optional[int] = 1,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0,
                 final_act: Optional[str] = None,
                 # Distinguish between link prediction and sequence prediction
                 predict_dst: bool = False,
                 # conv type Transformer
                 conv_type: str = 'TransformerConv',
                 separable: bool = False,
                 layernorm: bool = False
        ):
        super().__init__(num_nodes, edge_dim, memory_dim, time_dim, node_dim, num_iters, gnn_act, gnn_act_kwargs, epsilon, gamma, readout_hidden, readout_out, mean_delta_t, std_delta_t, init_time, final_act, predict_dst, conv_type)
        self.separable = separable
        self.layernorm = torch.nn.LayerNorm(memory_dim) if layernorm else None
        self.readout = TGBLinkPredictor(memory_dim)

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, dst = batch.src, batch.dst
        
        n_neg = batch.n_neg

        # Get updated memory of all nodes involved in the computation.
        z, last_update = self.memory(n_id)

        if hasattr(batch, 'x'):
            if len(batch.x.shape) == 3: # sequence classification case
                x = batch.x.squeeze(0)
            elif len(batch.x.shape) == 2: # link-based predictions
                x = batch.x
            else:
                raise ValueError(f"Unexpected node feature shape. Got {batch.x.shape}")
            z = torch.cat((z, x[n_id]), dim=-1)

        z = self.gnn(z, last_update, edge_index, t, msg) 
 
        if self.layernorm is not None:  
            z = self.layernorm(z)
 
        if self.final_act is not None:
            z = self.final_act(z)
        
        if self.predict_dst:
            assert n_neg is None
            pos_out = self.readout(z[id_mapper[dst]])
            neg_out = None   
        else:
            out = self.readout(z[id_mapper[src]], z[id_mapper[dst]])
            pos_out, neg_out = out[:-n_neg], out[-n_neg:]
    
        emb_src = z[id_mapper[src[:-n_neg]]]
        emb_pos_dst = z[id_mapper[dst[:-n_neg]]]

        return pos_out, neg_out, emb_src, emb_pos_dst
    
