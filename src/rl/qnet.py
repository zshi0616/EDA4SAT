import torch 
import deepgate as dg
import torch.nn as nn 
from progress.bar import Bar
from torch.nn import LSTM, GRU
from models.mlp import MLP

class Q_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.large_feature:
            dim_in = args.ckt_dim*2 + 6*64
        else:
            dim_in = args.ckt_dim*2 + 6
        self.mlp = MLP(dim_in, args.mlp_dim, args.n_action, \
            num_layer=args.mlp_layers, p_drop=0.2, act_layer='relu')
        
    def forward(self, obs):
        y_pred = self.mlp(obs)
        return y_pred
        
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
