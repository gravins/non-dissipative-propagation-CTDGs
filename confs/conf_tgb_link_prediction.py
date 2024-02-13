from utils import cartesian_product
from models import *


wiki = {  
    'embedding_dim': [256], 
    'lr': [1e-3, 3e-4, 1e-4, 3e-5],
    'wd': [0.], 
    'gnn_act': ['tanh'], 
    'num_iters': [1, 2, 3],
    'epsilon': [1.0, 0.5], 
    'gamma': [0.1],
    'sampler_size': [32],
    'final_act': [None, 'tanh'], 
    'separable': [False]
}

review = {  
    'embedding_dim': [256], 
    'lr': [1e-4],
    'wd': [0.], 
    'gnn_act': ['tanh'], 
    'num_iters': [1, 2],
    'epsilon': [1.0, 0.5], 
    'gamma': [0.1, 0.001],
    'sampler_size': [32],
    'final_act': [None, 'tanh'], 
    'separable': [False]
}

coin = {  
    'embedding_dim': [256], 
    'lr': [1e-4],
    'wd': [0.], 
    'gnn_act': ['tanh'], 
    'num_iters': [1],
    'epsilon': [1.0, 0.5], 
    'gamma': [0.1, 0.001],
    'sampler_size': [32],
    'final_act': [None, 'tanh'], 
    'separable': [False]
}

comment = {
    'embedding_dim': [256],
    'lr': [3e-4, 1e-4, 3e-5, 1e-5],
    'wd': [0],
    'gnn_act': ['tanh'],
    'num_iters': [1],
    'epsilon': [1.0],
    'gamma': [0.1],
    'sampler_size': [32],
    'final_act': [None],
    'separable': [False]
}
  
def get_CTAN_conf(confs, num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t):
    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'time_dim': params['embedding_dim'],
                'memory_dim': params['embedding_dim'],
                'gnn_act': params['gnn_act'],
                'num_iters': params['num_iters'],
                'epsilon': params['epsilon'],
                'gamma': params['gamma'],
                'readout_hidden': params['embedding_dim'],
                'readout_out': out_dim,
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time,
                'final_act': params['final_act'],
                'separable': params['separable']
            },
            'optim_params':{
                'lr': params['lr'],
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }

_ctan_fun_wiki = lambda num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t: get_CTAN_conf(wiki, num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t)
_ctan_fun_review = lambda num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t: get_CTAN_conf(review, num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t)
_ctan_fun_coin = lambda num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t: get_CTAN_conf(coin, num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t)
_ctan_fun_comment = lambda num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t: get_CTAN_conf(comment, num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t)

ctan_wiki = 'CTAN_tgb_wiki'
ctan_review = 'CTAN_tgb_review'
ctan_coin = 'CTAN_tgb_coin'
ctan_comment = 'CTAN_tgb_comment'

MODEL_CONFS = {
    ctan_wiki: (CTAN_tgb, _ctan_fun_wiki),
    ctan_review: (CTAN_tgb, _ctan_fun_review),
    ctan_coin: (CTAN_tgb, _ctan_fun_coin),
    ctan_comment: (CTAN_tgb, _ctan_fun_comment),
}
