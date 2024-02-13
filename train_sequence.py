import torch

from utils import scoring, optimizer_to, set_seed, dst_strategies, REGRESSION_SCORES, FakeScheduler
from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
from torch.utils.data import DataLoader
from datasets import get_dataset
import numpy as np
import datetime
import wandb
import time
import ray
import os


def train(train_data, model, optimizer, train_meta_loader, train_loaders, criterion, neighbor_loader, helper, device='cpu'):
    model.train()
    
    # Start with a fresh memory and an empty graph
    model.reset_memory()
    neighbor_loader.reset_state()

    for idx in train_meta_loader: # we iterate over batches of sequences. idx contains the indices of the sequences in the current batch
        optimizer.zero_grad()
        predictions, true_values = [], []
        for i in idx: # for each index we get the loader of the corresponding sequence
            i = i.int().item()
            loader = train_loaders[i]
            data = train_data[i]
            last_prediction = None
            for interaction in loader: # we iterate over the whole sequence
                interaction.to(device)
                src, pos_dst, t, msg = interaction.src, interaction.dst, interaction.t, interaction.msg

                n_id = torch.cat([src, pos_dst]).unique()
        
                edge_index = torch.empty(size=(2,0)).long()
                e_id = neighbor_loader.e_id[n_id]
                for _ in range(model.num_gnn_layers): # here we are retrieving the previous num_gnn_layers nodes in the sequence
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    e_id = e_id % data.msg.shape[0] # make e_id in the range [0, seq_len)

                helper[n_id] = torch.arange(n_id.size(0), device=device)
        
                # Model forward
                # NOTE: src_emb, pos_dst_emb are the embedding that will be saved in memory
                y_pred, _, src_emb, pos_dst_emb = model(batch=interaction, n_id=n_id, msg=data.msg[e_id].to(device),
                                                    t=data.t[e_id].to(device), edge_index=edge_index, id_mapper=helper)
                last_prediction = (y_pred.squeeze(0), interaction.y)
                
                # Update memory and neighbor loader with ground-truth state.
                model.update(src, pos_dst, t, msg, src_emb, pos_dst_emb)
                neighbor_loader.insert(src, pos_dst)

                # NOTE: we do not do model.detach_memory() here because otherwise we will compute the gradints only on 
                # the last event of the sequence rather than the computation on the entire sequence

            predictions.append(last_prediction[0])
            true_values.append(last_prediction[1])
        
        loss = criterion(torch.cat(predictions, dim=-1), torch.cat(true_values, dim=-1))
        loss.backward()
        #model.zero_grad_memory() # this allows us to not update the memory state, but still update the other parameters on the entire computation
        optimizer.step()
        model.detach_memory()


@torch.no_grad()
def eval(eval_data, model, meta_loader, eval_loaders, criterion, neighbor_loader, helper, return_predictions=False, 
         eval_name='eval', is_regression=False, wandb_log=False, device='cpu'):
    t0 = time.time()
    model.eval()

    y_pred_list, y_true_list, y_pred_confidence_list = [], [], []
    for idx in meta_loader: # we iterate over batches of sequences. idx contains the indices of the sequences in the current batch
        for i in idx: # for each index we get the loader of the corresponding sequence
            loader = eval_loaders[i]
            data = eval_data[i]
            last_prediction = None
            for interaction in loader: # we iterate over the whole sequence
                interaction.to(device)
                src, pos_dst, t, msg = interaction.src, interaction.dst, interaction.t, interaction.msg


                n_id = torch.cat([src, pos_dst]).unique()
                edge_index = torch.empty(size=(2,0)).long()
                e_id = neighbor_loader.e_id[n_id]
                for _ in range(model.num_gnn_layers): # here we are retrieving the previous num_gnn_layers nodes in the sequence
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                    e_id = e_id % data.msg.shape[0] # make e_id in the range [0, seq_len)


                helper[n_id] = torch.arange(n_id.size(0), device=device)

                # Model forward
                # NOTE: src_emb, pos_dst_emb are the embedding that will be saved in memory
                y_pred, _,  src_emb, pos_dst_emb = model(batch=interaction, n_id=n_id, msg=data.msg[e_id].to(device),
                                                    t=data.t[e_id].to(device), edge_index=edge_index, id_mapper=helper)
                last_prediction = (y_pred.squeeze(0), interaction.y)
      
                # Update memory and neighbor loader with ground-truth state.
                model.update(src, pos_dst, t, msg, src_emb, pos_dst_emb)
                neighbor_loader.insert(src, pos_dst)

            y_pred_confidence_list.append(last_prediction[0])
            y_true_list.append(last_prediction[1])
            y_pred_list.append((last_prediction[0].sigmoid() > 0.5).float())

    t1 = time.time()

    # Compute scores  
    y_true_list = torch.cat(y_true_list)
    y_pred_list = torch.cat(y_pred_list)
    y_pred_confidence_list = torch.cat(y_pred_confidence_list)
    scores = scoring(y_true_list, y_pred_list, y_pred_confidence_list, is_regression=is_regression)
    scores['loss'] = criterion(y_pred_confidence_list, y_true_list).item() 
    scores['time'] = datetime.timedelta(seconds=t1 - t0)

    true_values = (y_true_list, y_pred_list, y_pred_confidence_list) if return_predictions else None
    if wandb_log:
        for k, v in scores.items():
            if  k == 'confusion_matrix':
                continue
            else:
                wandb.log({f"{eval_name} {k}":v if k != 'time' else v.total_seconds()}, commit=False)
                
        _cm = wandb.plot.confusion_matrix(preds=y_pred_list.squeeze().numpy(),
                                          y_true=y_true_list.squeeze().numpy(),
                                          class_names=["negative", "positive"])
        wandb.log({f"conf_mat {eval_name}" : _cm}, commit='val' in eval_name or 'test' in eval_name)
        
    return scores, true_values


@ray.remote(num_cpus=int(os.environ.get('NUM_CPUS_PER_TASK', 1)), num_gpus=float(os.environ.get('NUM_GPUS_PER_TASK', 0.)))
def sequence_prediction(model_instance, conf):
    return sequence_prediction_single(model_instance, conf)


def sequence_prediction_single(model_instance, conf):
    if conf['wandb']:
        wandb.init(project=conf['data_name'], group=conf['model'], config=conf)

    # Set the configuration seed
    set_seed(conf['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data, _, _, _, _, _, _ = get_dataset(root=conf['data_dir'], 
                                      name=conf['data_name'], 
                                      seed=conf['exp_seed'])
    
    train_data, val_data, test_data = data.train_val_test_split(val_ratio=conf['split'][0], test_ratio=conf['split'][1])
    train_data = [d.to(device) for d in train_data]
    val_data = [d.to(device) for d in val_data]
    test_data = [d.to(device) for d in test_data]

    train_loaders = [TemporalDataLoader(train_seq, batch_size=1) for train_seq in train_data]
    val_loaders = [TemporalDataLoader(val_seq, batch_size=1) for val_seq in val_data]
    test_loaders = [TemporalDataLoader(test_seq, batch_size=1) for test_seq in test_data]

    train_meta_loader = DataLoader(torch.arange(0, len(train_loaders)), batch_size=conf['batch'], shuffle=True)
    val_meta_loader = DataLoader(torch.arange(0, len(val_loaders)), batch_size=conf['batch'], shuffle=False)
    test_meta_loader = DataLoader(torch.arange(0, len(test_loaders)), batch_size=conf['batch'], shuffle=False)

    neighbor_loader = LastNeighborLoader(data.num_nodes, size=conf['sampler']['size'], device=device)
    
    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    # Define model
    model = model_instance(**conf['model_params']).to(device)

    criterion = torch.nn.BCEWithLogitsLoss() if not conf['regression'] else REGRESSION_SCORES[conf['metric']] 
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['optim_params']['lr'], weight_decay=conf['optim_params']['wd'])

    if conf['lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10,
                                                                  mode='min' if conf['regression'] else 'max')
    else:
        lr_scheduler = FakeScheduler()

    history = []
    best_epoch = 0
    best_score = -np.inf if not conf['regression'] else np.inf
    isbest = lambda current, best, regression: current > best if not regression else current < best

    # Load previuos ckpt if exists
    path_save_best = os.path.join(conf['ckpt_path'], f'conf_{conf["conf_id"]}_seed_{conf["seed"]}.pt')
    if os.path.exists(path_save_best) and not conf['overwrite_ckpt']:
        # Load the existing checkpoint
        print(f'Loading {path_save_best}')
        ckpt = torch.load(path_save_best, map_location=device)
        best_epoch = ckpt['epoch']
        best_score = ckpt['best_score']
        history = ckpt['history']
        if ckpt['train_ended']:
            # The model was already trained, then return
            return ckpt['test_score'], ckpt['val_score'], ckpt['train_score'], ckpt['epoch'], conf, history
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_to(optimizer, device) # Map the optimizer to the current device
        lr_scheduler.load_state_dict(ckpt['scheduler'])
    model.to(device)
    
    epoch_times = []
    for e in range(best_epoch, conf['epochs']):
        t0 = time.time()
        if conf['debug']: print('Epoch {:d}:'.format(e))

        train(train_data=train_data, model=model, optimizer=optimizer, train_meta_loader=train_meta_loader, 
              train_loaders=train_loaders, criterion=criterion, neighbor_loader=neighbor_loader, helper=assoc, 
              device=device)
        
        model.reset_memory()
        neighbor_loader.reset_state()

        tr_scores, _ = eval(eval_data=train_data, model=model, meta_loader=train_meta_loader, 
                            eval_loaders=train_loaders, criterion=criterion, neighbor_loader=neighbor_loader, 
                            helper=assoc, eval_name='train_eval', is_regression=conf['regression'], 
                            wandb_log=conf['wandb'], device=device)
        
        if conf['reset_memory_eval']:
            model.reset_memory()

        vl_scores, vl_true_values = eval(eval_data=val_data, model=model, meta_loader=val_meta_loader, 
                                         eval_loaders=val_loaders, criterion=criterion,neighbor_loader=neighbor_loader, 
                                         helper=assoc, eval_name='val_eval', is_regression=conf['regression'], 
                                         wandb_log=conf['wandb'], device=device)

        history.append({
            'train': tr_scores,
            'val': vl_scores
        })

        lr_scheduler.step(vl_scores[conf['metric']])

        if len(history) == 1 or isbest(vl_scores[conf['metric']], best_score, conf['regression']):
            best_score = vl_scores[conf['metric']]
            best_epoch = e
            torch.save({
                'train_ended': False,
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'best_score': best_score,
                'loss': (tr_scores['loss'], vl_scores['loss'], None),
                'tr_scores': tr_scores,
                'vl_scores': vl_scores,
                'true_values': (None, vl_true_values, None),
                'history': history
            }, path_save_best)

        if conf['debug']: print(f'\ttrain :{tr_scores}\n\tval :{vl_scores}')
        epoch_times.append(time.time()-t0)

        if conf['debug'] or (conf['verbose'] and e % conf['patience'] == 0): 
            print(f'Epoch {e}: {np.mean(epoch_times)} +/- {np.std(epoch_times)} seconds per epoch') 

        if e - best_epoch > conf['patience']:
            break

    # Evaluate on test
    if conf['debug']: print('Loading model at epoch {}...'.format(best_epoch))
    ckpt = torch.load(path_save_best, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    ckpt['test_score'] = {}
    ckpt['val_score'] = {}
    ckpt['train_score'] = {}
    ckpt['true_values'] = {}
    ckpt['loss'] = {}

    strategy = dst_strategies[0] # Fake value. For regression tasks we do not use strategies

    model.reset_memory()
    neighbor_loader.reset_state()
    
    tr_scores, tr_true_values = eval(eval_data=train_data, model=model, meta_loader=train_meta_loader, 
                                     eval_loaders=train_loaders, criterion=criterion, neighbor_loader=neighbor_loader, 
                                     helper=assoc, return_predictions=True, eval_name='train_eval', 
                                     is_regression=conf['regression'], wandb_log=conf['wandb'], device=device)
        
    if conf['reset_memory_eval']:
        model.reset_memory()

    vl_scores, vl_true_values = eval(eval_data=val_data, model=model, meta_loader=val_meta_loader, 
                                     eval_loaders=val_loaders, criterion=criterion, neighbor_loader=neighbor_loader, 
                                     helper=assoc, return_predictions=True, eval_name='val_eval', 
                                     is_regression=conf['regression'], wandb_log=conf['wandb'], device=device)
        
    if conf['reset_memory_eval']:
        model.reset_memory()

    ts_scores, ts_true_values = eval(eval_data=test_data, model=model, meta_loader=test_meta_loader, 
                                    eval_loaders=test_loaders, criterion=criterion, neighbor_loader=neighbor_loader, 
                                    helper=assoc, return_predictions=True, eval_name='test_eval', 
                                    is_regression=conf['regression'], wandb_log=conf['wandb'], device=device)

    ckpt['test_score'][strategy] = ts_scores
    ckpt['val_score'][strategy] = vl_scores
    ckpt['train_score'][strategy] = tr_scores
    ckpt['true_values'][strategy] = (tr_true_values, vl_true_values, ts_true_values)
    ckpt['loss'][strategy] = (tr_scores['loss'], vl_scores['loss'], ts_scores['loss'])

    ckpt['train_ended'] = True
    torch.save(ckpt, path_save_best)

    history = ckpt['history'] if conf['log'] else None
    conf['model size'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return ckpt['test_score'], ckpt['val_score'], ckpt['train_score'], ckpt['epoch'], conf, history
