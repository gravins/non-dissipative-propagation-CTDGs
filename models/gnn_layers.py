import torch
from torch_geometric.nn import AntiSymmetricConv, TransformerConv
from typing import Callable, Optional, Dict, Union, Any
from torch_geometric.nn.models.tgn import TimeEncoder
from .layers import NormalLinear
from typing import Callable


class GraphAttentionEmbedding(torch.nn.Module):
    ''' GNN layer of the original TGN model '''

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: Callable,
                 mean_delta_t: float = 0., std_delta_t: float = 1., 
                 # conv type Transformer
                 conv_type: str = 'TransformerConv'):
        
        assert conv_type in ['TransformerConv']

        super().__init__()
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)
        if conv_type == 'TransformerConv':
            self.conv = TransformerConv(in_channels, out_channels, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)
        else:
            raise NotImplementedError()

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t =  (last_update[edge_index[0]] - t).abs()
        rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t # delta_t normalization
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))

        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class JodieEmbedding(torch.nn.Module):
    def __init__(self, out_channels: int,
                 mean_delta_t: float = 0., std_delta_t: float = 1.):
        super().__init__()
        
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.projector = NormalLinear(1, out_channels)

    def forward(self, x, last_update, t):
        rel_t = (last_update - t).abs()
        if rel_t.shape[0] > 0:
            rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t # delta_t normalization
            return x * (1 + self.projector(rel_t.view(-1, 1).to(x.dtype))) 
    
        return x


class CTANEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int, node_dim: int, msg_dim: int, time_dim: int,
                 num_iters: int = 1, 
                 act: Union[str, Callable, None] = 'tanh',
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 epsilon: float = 0.1, 
                 gamma: float = 0.1,
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 # conv type Transformer
                 conv_type: str = 'TransformerConv'):
        
        assert conv_type in ['TransformerConv']

        super().__init__()
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t

        self.out_channels = in_channels

        self.enc_x = torch.nn.Linear(in_channels+node_dim, in_channels)

        if conv_type == 'TransformerConv':
            phi = TransformerConv(in_channels, in_channels, edge_dim=msg_dim+time_dim, root_weight=False)
        else:
            raise NotImplementedError()

        self.aconv = AntiSymmetricConv(in_channels, phi, num_iters, epsilon, gamma, act=act, act_kwargs=act_kwargs)
        self.time_enc = TimeEncoder(time_dim)
        
    def reset_parameters(self):
        self.time_enc.reset_parameters()
        self.aconv.reset_parameters()

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = (last_update[edge_index[0]] - t).abs()
        rel_t = ((rel_t - self.mean_delta_t) / self.std_delta_t).to(x.dtype) # delta_t normalization
        enc_x = self.enc_x(x)
        edge_attr = torch.cat([msg, self.time_enc(rel_t)], dim=-1)
        return self.aconv(enc_x, edge_index, edge_attr=edge_attr)
    
