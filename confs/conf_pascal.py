from utils import cartesian_product
from models import *
import copy

shared_params = {
    'time_dim': [1],
    'lr': [3e-4], 
    'wd': [1e-5], 
    'gnn_act': ['tanh'],
    'sampler_size': [5],
    'num_gnn_layers': [5, 3, 1]
}

def get_TGN_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t):
    confs = copy.deepcopy(shared_params)
    confs["embedding_dim"] = [21,10]

    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'memory_dim': params['embedding_dim'],
                'time_dim': params['time_dim'],
                'gnn_hidden_dim': [params['embedding_dim']] * params['num_gnn_layers'],
                'gnn_act': params['gnn_act'],
                'readout_hidden': max(1, params['embedding_dim'] // 2),
                'readout_out': out_dim,
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time
            },
            'optim_params':{
                'lr': params['lr'],
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }


def get_DyRep_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t):
    confs = {
        'lr': shared_params['lr'],
        'wd': shared_params['wd'],
        'non_linearity': shared_params['gnn_act'],
        'sampler_size': shared_params['sampler_size']
    }

    confs["embedding_dim"] = [74, 37]

    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'memory_dim': params['embedding_dim'],
                'non_linearity': params['non_linearity'],
                'readout_hidden': max(1, params['embedding_dim'] // 2),
                'readout_out': out_dim,
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time
            },
            'optim_params':{
                'lr': params['lr'],
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }


def get_JODIE_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t):
    confs = {
        'time_dim': shared_params['time_dim'],
        'lr': shared_params['lr'],
        'wd': shared_params['wd'],
        'non_linearity': shared_params['gnn_act'],
        'sampler_size': shared_params['sampler_size']
    }

    confs["embedding_dim"] = [97, 48]

    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'memory_dim': params['embedding_dim'],
                'time_dim': params['time_dim'],
                'non_linearity': params['non_linearity'],
                'readout_hidden': max(1, params['embedding_dim'] // 2),
                'readout_out': out_dim,
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time
            },
            'optim_params':{
                'lr': params['lr'],
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }


def get_TGAT_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t):
    confs = copy.deepcopy(shared_params)
    confs['embedding_dim'] = [24, 12]

    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'time_dim': params['time_dim'],
                'gnn_hidden_dim': [params['embedding_dim']] * params['num_gnn_layers'],
                'gnn_act': params['gnn_act'],
                'readout_hidden': max(1, params['embedding_dim'] // 2),
                'readout_out': out_dim,
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time
            },
            'optim_params':{
                'lr': params['lr'],
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }



def get_CTAN_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t):
    confs = {
        'time_dim': shared_params['time_dim'],
        'lr': shared_params['lr'],
        'wd': shared_params['wd'],
        'gnn_act': shared_params['gnn_act'],
        'num_iters': shared_params['num_gnn_layers'],
        'epsilon': [0.5, 0.2, 0.1, 0.01],
        'gamma': [0.5, 0.2, 0.1, 0.01],
        'sampler_size': shared_params['sampler_size'],
        'final_act': [None, 'tanh']
    }

    confs["embedding_dim"] = [74, 37]

    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'time_dim': params['time_dim'],
                'memory_dim': params['embedding_dim'],
                'gnn_act': params['gnn_act'],
                'num_iters': params['num_iters'],
                'epsilon': params['epsilon'],
                'gamma': params['gamma'],
                'readout_hidden': max(1, params['embedding_dim'] // 2),
                'readout_out': out_dim,
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time,
                'final_act': params['final_act'],
            },
            'optim_params':{
                'lr': params['lr'],
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }


_tgn_fun = lambda num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t: get_TGN_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t)
_dyrep_fun = lambda num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t: get_DyRep_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t)
_jodie_fun = lambda num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t: get_JODIE_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t)
_tgat_fun = lambda num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t: get_TGAT_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t)
_ctan_fun = lambda num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t: get_CTAN_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t)

tgn, dyrep, jodie, tgat = 'TGN', 'DyRep', 'JODIE', 'TGAT'
ctan = 'CTAN'

MODEL_CONFS = {
    tgn: (TGN, _tgn_fun),
    dyrep: (DyRep, _dyrep_fun),
    jodie: (JODIE, _jodie_fun),
    tgat: (TGAT, _tgat_fun),
    ctan: (CTAN, _ctan_fun)
}
