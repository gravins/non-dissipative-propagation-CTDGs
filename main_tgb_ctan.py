import matplotlib
matplotlib.use('Agg')

import torch 

from train_link import tgb_link_prediction, tgb_link_prediction_single
from utils import (set_seed, SCORE_NAMES, compute_stats, MRR)
from datasets import get_dataset, DATA_NAMES
from collections import defaultdict
import matplotlib.pyplot as plt
from confs import MODEL_CONFS
import pandas as pd 
import numpy as np 
import argparse
import datetime
import pickle
import utils
import time
import tqdm
import ray
import os
import gc
torch.autograd.set_detect_anomaly(True)

def compute_row(test_score, val_score, train_score, best_epoch, res_conf):
    # This function computes one row of the CSV file containing the results of each configuration
    row = {}

    for label, score_dict in [('test', test_score), ('val', val_score), ('train', train_score)]:
        print(label, score_dict)
        for k, v in score_dict.items():
            row[f'{label}_{k}'] = v

    for k in res_conf.keys():
        if isinstance(res_conf[k], dict):
            for sk in res_conf[k]:
                row[f'{k}_{sk}'] = res_conf[k][sk]
        else:
            row[k] = res_conf[k]
    row.update({f'best_epoch': best_epoch})
    return row


def save_log(history, log_path, conf):
    # Given a configuration, the function makes the plot of each metric wrt epochs
    res = {
        'train': defaultdict(list), 
        'val': defaultdict(list)
    }
    for epoch in range(len(history)):
        for mode in ['train', 'val']:
            for k, v in history[epoch][mode].items():
                if 'confusion' in k: 
                    continue
                res[mode][k].append(
                    v.total_seconds() if 'time' in k else v if v != np.nan else 1e13
                )

    for k in res['train'].keys():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(res['train'][k], label='tr')
        ax.plot(res['val'][k], label='vl')
        ax.set(
            xlabel='Epochs', 
            ylabel=k
        )
        title=str(conf['model_params']) + ' ' + str(conf['optim_params']) + ' ' + str(conf['sampler'])
        ax.set_title(title, loc='center', wrap=True)
        plt.legend(loc='best')
        plt.tight_layout()
        fig.savefig(os.path.join(log_path, f"{conf['conf_id']}_{k}.png"))
        plt.close()


def aggregate_res(df, args, is_assessment=False):
    # Aggregate results over multiple runs and sort them by best val score
    aggregated_df = []
    for conf_id, gdf in df.groupby('conf_id'):
        if (args.num_runs == 1 and is_assessment == False) or (args.num_final_runs == 1 and is_assessment):
            row = gdf.iloc[0]
        else:
            row = {}
            for k in gdf.columns:
                if k == 'seed': 
                    row[k] = gdf[k].values 
                if 'test' in k or 'val' in k or 'train' in k or k == 'best_epoch':
                    row[f'{k}_mean'] = gdf[k].values.mean() if 'confusion_matrix' in k else gdf[k].mean()
                    row[f'{k}_std'] = gdf[k].values.std() if 'confusion_matrix' in k else gdf[k].std()
                else:
                    row[k] = gdf.iloc[0][k]
        aggregated_df.append(row)
    aggregated_df = pd.DataFrame(aggregated_df)

    col_name = (f'val_{args.metric}' 
                if (args.num_runs == 1 and is_assessment == False) or (args.num_final_runs == 1 and is_assessment) 
                else f'val_{args.metric}_mean')
    aggregated_df = aggregated_df.sort_values(col_name, ascending=False)
    return aggregated_df


def wait_and_collect(ray_ids, df, args, partial_res_pkl, partial_res_csv, pbar, log_path=None):
    # Collect the results of each evaluated configuration
    done_id, ray_ids = ray.wait(ray_ids)
    test_score, val_score, train_score, best_epoch, res_conf, history = ray.get(done_id[0])
    df.append(compute_row(test_score, val_score, train_score, best_epoch, res_conf))
    if args.log: save_log(history, log_path, res_conf)
    pickle.dump(df, open(partial_res_pkl, 'wb'))
    pd.DataFrame(df).to_csv(partial_res_csv)
    pbar.update(1)
    gc.collect()
    return ray_ids, df, pbar
    

def run_configs(conf_list, args, num_runs, model_instance, train_config_single, train_config, result_path, ckpt_path, 
                partial_res_pkl, partial_res_csv, cpus_per_task, gpus_per_task, log_path=None, fixed_conf_id=None):
    # Train a set of configuration and collect their results
    num_conf = len(conf_list)
    pbar = tqdm.tqdm(total= num_conf*num_runs)
    df = []
    ray_ids = []
    for conf_id, conf in enumerate(conf_list):
        for i in range(num_runs):
            conf.update({
                'conf_id': conf_id if fixed_conf_id is None else fixed_conf_id,
                'seed': i,
                'result_path': result_path,
                'ckpt_path': ckpt_path,
            })
            conf.update(vars(args))

            if args.debug:
                (test_score, val_score, train_score, 
                    best_epoch, res_conf, history) = train_config_single(model_instance, conf)

                df.append(compute_row(test_score, val_score, train_score, best_epoch, res_conf))
                if args.log: save_log(history, log_path, res_conf)
                pickle.dump(df, open(partial_res_pkl, 'wb'))
                pbar.update(1)
            else:
                opt = { 
                    'num_cpus': cpus_per_task, 
                    'num_gpus': gpus_per_task, 
                    'memory': args.memory_per_task, 
                    'scheduling_strategy': args.scheduling_strategy
                }
                ray_ids.append(train_config.options(**opt).remote(model_instance, conf))
            
            if args.parallelism is not None:
                while len(ray_ids) > args.parallelism:
                    ray_ids, df, pbar = wait_and_collect(ray_ids = ray_ids,
                                                         df = df,
                                                         args = args,
                                                         partial_res_pkl = partial_res_pkl, 
                                                         partial_res_csv = partial_res_csv,
                                                         log_path = log_path,
                                                         pbar = pbar)

            gc.collect()
    
    while len(ray_ids):
        ray_ids, df, pbar = wait_and_collect(ray_ids = ray_ids,
                                             df = df,
                                             args = args,
                                             partial_res_pkl = partial_res_pkl, 
                                             partial_res_csv = partial_res_csv,
                                             log_path = log_path,
                                             pbar = pbar)
    return df


if __name__ == "__main__":
    t0 = time.time()
 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', help='The path to the directory where data files are stored.', default='./DATA')
    parser.add_argument('--data_name', help='The data name.', default=DATA_NAMES[0], choices=DATA_NAMES)
    parser.add_argument('--save_dir', help='The path to the directory where checkpoints/results are stored.', default='./RESULTS/')
    parser.add_argument('--model', help='The model name.', default=list(MODEL_CONFS)[0], choices=MODEL_CONFS.keys())
    parser.add_argument('--num_runs', help='The number of random initialization per conf.', default=5, type=int)
    parser.add_argument('--num_final_runs', help='The number of random initialization for the best selected conf.', default=5, type=int)
    parser.add_argument('--epochs', help='The number of epochs.', default=200, type=int) 
    parser.add_argument('--batch', help='The batch_size.', default=200, type=int)
    parser.add_argument('--patience', help='The early stopping patience, ie train is stopped if no score improvement after X epochs.', default=50, type=int)
    parser.add_argument('--exp_seed', help='The experimental seed.', default=42, type=int)
    parser.add_argument('--metric', help='The optimized metric.', default=MRR, choices=list(SCORE_NAMES)) 
    parser.add_argument('--split', help='(val_ratio, test_ratio) split ratios.', nargs=2, default=[.15, .15])
      
    parser.add_argument('--lr_scheduler', help='Use a learning rate scheduler.', action='store_true')
    parser.add_argument('--lr_scheduler_patience', help='The scheduler patience.', type=int, default=15)
    
    parser.add_argument('--validate_every', help='Epochs interval to run validation.', default=5, type=int)
    parser.add_argument('--validation_subsample', help='In validation, how much to subsample the negative destinations.', type=int, default=None)
    parser.add_argument('--validation_batched', help='In validation, whether to batch samples before forward.', action='store_true')

    parser.add_argument('--cluster', help='Experiments run on a cluster.', action='store_true')
    parser.add_argument('--slurm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--parallelism', help='The degree of parallelism, ie, maximum number of parallel jobs.', default=None, type=int)
    parser.add_argument('--overwrite_ckpt', help='Overwrite checkpoint.', action='store_true')
    parser.add_argument('--verbose', help='Every <patience> epochs it prints the average time to compute an epoch.', action='store_true')
    parser.add_argument('--log', help='Print model history without wandb.', action='store_true')

    parser.add_argument('--num_cpus', help='The number of total available cpus.', default=2, type=int)
    parser.add_argument('--num_gpus', help='The number of total available gpus.', default=0, type=int)
    parser.add_argument('--num-cpus-per-task', help='The number of cpus available for each model config.', default=-1, type=int)
    parser.add_argument('--num-gpus-per-task', help='The number of gpus available for each model config.', default=-0., type=float)
    parser.add_argument(
        '--memory_per_task',
        help='The amount of memory in bytes required for each model config, used by Ray to schedule new tasks. \
              Please refer to https://docs.ray.io/en/releases-2.4.0/ray-core/scheduling/memory-management.html#memory-aware-scheduling',
              default=None,
              type=int)
    parser.add_argument('--scheduling_strategy', help='Scheduling strategy employed by Ray to schedule new configurations. Please refer to https://docs.ray.io/en/releases-2.4.0/ray-core/scheduling/index.html', default='SPREAD', choices=['DEFAULT', 'SPREAD'])

    args = parser.parse_args()

    assert args.cluster or args.debug

    # Set resources
    # args.memory_per_task = int(args.memory_per_task) if str.isdecimal(args.memory_per_task) else args.memory_per_task
    cpus_per_task = int(os.environ.get('NUM_CPUS_PER_TASK', -1))
    gpus_per_task = float(os.environ.get('NUM_GPUS_PER_TASK', -1.))
    cpus_per_task = args.num_cpus_per_task if cpus_per_task < 0 else cpus_per_task
    gpus_per_task = args.num_gpus_per_task if gpus_per_task < 0 else gpus_per_task
    assert cpus_per_task > -1, 'You must define the number of CPUS per task, by setting --num_cpus_per_task or exporting the variable NUM_CPUS_PER_TASK'
    assert gpus_per_task > -1, 'You must define the number of GPUS per task, by setting --num_gpus_per_task or exporting the variable NUM_GPUS_PER_TASK'

    # Set the experimental seed
    set_seed(args.exp_seed)

    cpus = os.environ.get('NUM_CPUS', None) or args.num_cpus
    gpus = os.environ.get('NUM_GPUS', None) or args.num_gpus
    print(f"Local resources: CPUS: {int(cpus)}, GPUS: {int(gpus)}")
 
    if not args.debug:
        cpus = os.environ.get('NUM_CPUS', None)
        gpus = os.environ.get('NUM_GPUS', None)
        cpus = int(cpus) if cpus is not None else args.num_cpus
        gpus = int(gpus) if gpus is not None else args.num_gpus
        utils.setup_cluster(cpus, gpus, args.slurm, pack_tgb=False)
    
    # Create all dirs and path used in the experiments
    args.save_dir = os.path.abspath(args.save_dir)
    args.data_dir = os.path.join(os.path.abspath(args.data_dir))
    result_path = os.path.join(args.save_dir, args.data_name, args.model)
    ckpt_path = os.path.join(result_path, 'ckpt')
    log_path = os.path.join(result_path, 'log')

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    partial_res_pkl = os.path.join(result_path, 'partial_results.pkl')
    partial_res_csv = os.path.join(result_path, 'partial_results.csv')
    final_res_csv = os.path.join(result_path, 'model_selection_results.csv')

    print(f'\n{args}\n')
    print(f'Data dir: {args.data_dir}')
    print(f'Results dir: {result_path}')
    print(f'Checkpoints dir: {ckpt_path}')
 
    # Load the data 
    data, num_nodes, edge_dim, node_dim, out_dim, init_time, dataset = get_dataset(root=args.data_dir, name=args.data_name,  seed=args.exp_seed)
    
    # Compute statistics on the dataset's timestamps to perform delta_t normalization
    stat_path = os.path.join(args.data_dir, args.data_name.replace("-", "_").lower(), 'delta_t_stats.pkl')
    if os.path.exists(stat_path):
        mean_delta_t, std_delta_t = pickle.load(open(stat_path, 'rb')) 
    else:
        mean_delta_t, std_delta_t = compute_stats(data, args.split, init_time, sequence_prediction=False)
        pickle.dump((mean_delta_t, std_delta_t), open(stat_path, 'wb'))
    
    # Run model selection 
    model_instance, get_conf = MODEL_CONFS[args.model]
    conf_list = list(get_conf(num_nodes, edge_dim, node_dim, out_dim, init_time, mean_delta_t, std_delta_t))
    df = run_configs(
        conf_list = conf_list,
        args = args, 
        num_runs = args.num_runs,
        model_instance = model_instance,
        train_config_single = tgb_link_prediction_single ,
        train_config = tgb_link_prediction,
        result_path = result_path,
        ckpt_path = ckpt_path,
        partial_res_pkl = partial_res_pkl,
        partial_res_csv = partial_res_csv,
        cpus_per_task = cpus_per_task,
        gpus_per_task = gpus_per_task,
        log_path = log_path if args.log else None
    )
    
    df = pd.DataFrame(df)

    # Aggregate results over multiple runs and sort them by best val score
    aggregated_df = aggregate_res(df, args)
    aggregated_df.to_csv(final_res_csv)
    print(aggregated_df.iloc[0].to_string())

    t1 = time.time()
    print(f'Main ended in {datetime.timedelta(seconds=t1 - t0)}')
