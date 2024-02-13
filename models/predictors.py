import torch
from typing import Optional

class TGBLinkPredictor(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, in_channels)
        self.lin_dst = torch.nn.Linear(in_channels, in_channels)
        self.lin_final = torch.nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)
    

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None, out_channels: int = 1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.lin_src = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_dst = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_final = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)
    

class SequencePredictor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None, out_channels: int = 1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.lin = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_final = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, z_dst):
        h = self.lin(z_dst)
        h = h.relu()
        return self.lin_final(h)