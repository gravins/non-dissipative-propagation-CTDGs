import torch

from utils import get_node_sets, scoring, optimizer_to, set_seed, dst_strategies, REGRESSION_SCORES, FakeScheduler
from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import TemporalData
from datasets import get_dataset
import negative_sampler
import numpy as np
import datetime
import time
import ray
import os
from tqdm import tqdm
from typing import List
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.evaluate import Evaluator
from confs.conf_link_prediction import edgebank
import wandb



@torch.no_grad()
def tgb_test(data, model, loader, neg_sampler,
             neighbor_loader, split_mode, helper,
             evaluator, metric, validation_subsample=None,
             batched_val=False, device="cpu", pbar=False):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    model.eval()
 
    breakf = False

    perf_list = []
    iterator = loader if not pbar else tqdm(loader)
    
    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    for jj, batch in enumerate(iterator):
        if breakf and jj > 5: break
        batch = batch.to(device)

        pos_src, pos_dst, pos_t, pos_msg = (
            batch.src,
            batch.dst,
            batch.t,
            batch.msg,
        )

        if split_mode in ["val_fast"]:

            neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode="val")
            src_embs, pos_dst_embs = [], []

            for idx, neg_dst in enumerate(neg_batch_list):
                # separate positive dst and negative dst forward loop because dimensions don't match
                # in this case: there is one positive but N negative dst
                if validation_subsample is not None:
                    #to_sample = 20 # min(20, int(len(neg_dst)*validation_subsample))
                    neg_dst = np.random.choice(neg_dst, size=validation_subsample, replace=True)

                _src = torch.full((1 + len(neg_dst),), pos_src[idx], device=device)
                _pos_dst = pos_dst[idx].view(1)
                _neg_dst = torch.tensor(neg_dst, device=device)

                original_n_id = torch.cat([pos_src[idx].view(1), _pos_dst, _neg_dst]).unique()

                n_id = original_n_id
                edge_index = torch.empty(size=(2,0)).long()
                e_id = neighbor_loader.e_id[n_id]
                for _ in range(model.num_gnn_layers):
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                helper[n_id] = torch.arange(n_id.size(0), device=device)
                
                # ### block ###
                _batch = TemporalData(
                    src = _src,
                    dst = torch.cat([_pos_dst, _neg_dst], -1),
                    x = batch.x,
                    n_neg=len(_neg_dst)
                )

                pos_out, neg_out, src_emb, pos_dst_emb = model(
                    batch=_batch, n_id=n_id, msg=data.msg[e_id].to(device),
                    t=data.t[e_id].to(device), edge_index=edge_index, id_mapper=helper
                )
                # compute MRR
                input_dict = {
                    "y_pred_pos": np.array(pos_out.squeeze(dim=-1).cpu()),
                    "y_pred_neg": np.array(neg_out.squeeze(dim=-1).cpu()),
                    "eval_metric": [metric],
                }
                perf_list.append(evaluator.eval(input_dict)[metric])

                src_embs.append(src_emb)
                pos_dst_embs.append(pos_dst_emb) 
            
            src_emb = torch.cat(src_embs, 0)
            pos_dst_emb = torch.cat(pos_dst_embs, 0)

        elif split_mode in ["test", "val"]:
            neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

            src_embs = []
            pos_dst_embs = []

            for idx, neg_dst in enumerate(neg_batch_list):
                # separate positive dst and negative dst forward loop because dimensions don't match
                # in this case: there is one positive but N negative dst
                
                src = torch.full((1 + len(neg_dst),), pos_src[idx], device=device)
                
                neg_dst = torch.tensor(neg_dst, dtype=torch.long, device=device)

                original_n_id = torch.cat([src, pos_dst, neg_dst]).unique()
 
                n_id = original_n_id
                edge_index = torch.empty(size=(2,0)).long()
                e_id = neighbor_loader.e_id[n_id]
                for _ in range(model.num_gnn_layers):
                    n_id, edge_index, e_id = neighbor_loader(n_id)
                helper[n_id] = torch.arange(n_id.size(0), device=device)

                neg_src = torch.full((len(neg_dst), ), pos_src[idx], device=device)  # [N,]
                 
                # ### block ###
                _batch = TemporalData(
                    src = torch.cat([pos_src[idx].view(1), neg_src], -1),
                    dst = torch.cat([pos_dst[idx].view(1), neg_dst], -1),
                    x = batch.x,
                    n_neg=len(neg_dst)
                )

                pos_out, neg_out, src_emb, pos_dst_emb = model(
                    batch=_batch, n_id=n_id, msg=data.msg[e_id].to(device),
                    t=data.t[e_id].to(device), edge_index=edge_index, id_mapper=helper
                )

                # compute MRR
                input_dict = {
                    "y_pred_pos": np.array(pos_out.squeeze(dim=-1).cpu()),
                    "y_pred_neg": np.array(neg_out.squeeze(dim=-1).cpu()),
                    "eval_metric": [metric],
                }
                perf_list.append(evaluator.eval(input_dict)[metric])

                src_embs.append(src_emb)
                pos_dst_embs.append(pos_dst_emb)
            src_emb = torch.cat(src_embs, 0)
            pos_dst_emb = torch.cat(pos_dst_embs, 0)
        elif split_mode == "train":
            # Sample negative destination nodes.
            neg_dst = torch.randint(
                min_dst_idx,
                max_dst_idx + 1,
                (pos_src.size(0),),
                dtype=torch.long,
                device=device,
            )
            original_n_id = torch.cat([pos_src, pos_dst, neg_dst]).unique()

            batch.n_neg = len(neg_dst)
            batch.src = torch.cat([pos_src, pos_src], -1).to(device)
            batch.dst = torch.cat([pos_dst, neg_dst], -1).to(device)

            n_id = original_n_id
            edge_index = torch.empty(size=(2,0)).long()
            e_id = neighbor_loader.e_id[n_id]
            for _ in range(model.num_gnn_layers):
                n_id, edge_index, e_id = neighbor_loader(n_id)
            helper[n_id] = torch.arange(n_id.size(0), device=device)

            pos_out, neg_out, src_emb, pos_dst_emb = model(
                batch=batch, n_id=n_id, msg=data.msg[e_id].to(device),
                t=data.t[e_id].to(device), edge_index=edge_index, 
                id_mapper=helper
            )

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array(pos_out.squeeze(dim=-1).cpu()),
                "y_pred_neg": np.array(neg_out.squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            } 
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.
        model.update(pos_src, pos_dst, pos_t, pos_msg, src_emb, pos_dst_emb)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(perf_list).mean())

    return {"mrr": perf_metrics}, None


def tgb_train(data, model, optimizer, scheduler,
              train_loader, criterion,
              neighbor_loader, helper,
              device='cpu', pbar=False):
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """

    model.train()
    data.msg = data.msg.to(device)
    data.t = data.t.to(device)

    # Start with a fresh memory and an empty graph
    model.reset_memory()
    neighbor_loader.reset_state()

    losses = []
    iterator = train_loader if not pbar else tqdm(train_loader)

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    breakf = False  

    for jj, batch in enumerate(iterator):
        if breakf and jj > 5: break
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )
        original_n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        
        batch.n_neg = len(neg_dst)
        batch.src = torch.cat([src, src], -1).to(device)
        batch.dst = torch.cat([pos_dst, neg_dst], -1).to(device)

        n_id = original_n_id
        edge_index = torch.empty(size=(2,0)).long()
        e_id = neighbor_loader.e_id[n_id]
        for _ in range(model.num_gnn_layers):
            n_id, edge_index, e_id = neighbor_loader(n_id)
        helper[n_id] = torch.arange(n_id.size(0), device=device)

        pos_out, neg_out, src_emb, pos_dst_emb = model(batch=batch, n_id=n_id, msg=data.msg[e_id].to(device),
                                                       t=data.t[e_id].to(device), edge_index=edge_index, id_mapper=helper)


        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))
    
        losses.append(loss.item())

        # Update memory and neighbor loader with ground-truth state.
        model.update(src, pos_dst, t, msg, src_emb, pos_dst_emb)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()

        if pbar and jj % 50 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:    
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            print(f"Loss: {np.array(losses[-20:]).mean():.5f}, LR: {optimizer.param_groups[0]['lr']:.5f}, Scheduler LR: {scheduler.get_last_lr()[0]:.5f}, Grad Norm: {total_norm:.5f}")

        # clip_grad_value_(model.parameters(), 0.1)

        optimizer.step()
        model.detach_memory()

    return losses


def train(data, model, optimizer, train_loader, criterion, neighbor_loader, helper, train_neg_sampler=None, 
          requires_grad=True, device='cpu', pbar=False):
    model.train()
    data.msg = data.msg.to(device)
    data.t = data.t.to(device)

    # Start with a fresh memory and an empty graph
    model.reset_memory()
    neighbor_loader.reset_state()

    losses = []

    iterator = train_loader if not pbar else tqdm(train_loader)

    for batch in iterator:
        batch = batch.to(device)
        optimizer.zero_grad()
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        if train_neg_sampler is None:
            # NOTE: When the train_neg_sampler is None we are doing link regression or multiclass classification
            original_n_id = torch.cat([src, pos_dst]).unique()
        else:
            # NOTE: We are doing link prediction
            # Sample negative destination nodes.
            neg_dst = train_neg_sampler.sample(src).to(device)
            original_n_id = torch.cat([src, pos_dst, neg_dst]).unique()
            batch.neg_dst = neg_dst

        n_id = original_n_id
        edge_index = torch.empty(size=(2,0)).long()
        e_id = neighbor_loader.e_id[n_id]
        for _ in range(model.num_gnn_layers):
            n_id, edge_index, e_id = neighbor_loader(n_id)

        helper[n_id] = torch.arange(n_id.size(0), device=device)
        
        # Model forward
        # NOTE: src_emb, pos_dst_emb are the embedding that will be saved in memory
        pos_out, neg_out, src_emb, pos_dst_emb = model(batch=batch, n_id=n_id, msg=data.msg[e_id].to(device),
                                                       t=data.t[e_id].to(device), edge_index=edge_index, id_mapper=helper)
        
        if train_neg_sampler is None:
            loss = criterion(pos_out, batch.y)
        else:
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

        losses.append(loss.item())

        # Update memory and neighbor loader with ground-truth state.
        model.update(src, pos_dst, t, msg, src_emb, pos_dst_emb)
        neighbor_loader.insert(src, pos_dst)
        
        if requires_grad:
            loss.backward()
            optimizer.step()
        model.detach_memory()
    
    return losses


@torch.no_grad()
def eval(data, model, loader, criterion, neighbor_loader, helper, neg_sampler=None, 
         eval_seed=12345, regression=False, multiclass=False, return_predictions=False,
         device='cpu', eval_name='eval', wandb_log=False, pbar=False):
    t0 = time.time()
    model.eval()
    data.msg = data.msg.to(device)
    data.t = data.t.to(device)

    y_pred_list, y_true_list, y_pred_confidence_list = [], [], []

    iterator = loader if not pbar else tqdm(loader)

    for batch in iterator:
        batch = batch.to(device)
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        if neg_sampler is None:
            # NOTE: When the neg_sampler is None we are doing link regression or multiclass classification
            original_n_id = torch.cat([src, pos_dst]).unique()
        else:
            # NOTE: We are doing link prediction
            # Sample negative destination nodes
            neg_dst = neg_sampler.sample(src, eval=True, eval_seed=eval_seed).to(device) # Ensure deterministic sampling across epochs
            original_n_id = torch.cat([src, pos_dst, neg_dst]).unique()
            batch.neg_dst = neg_dst

        n_id = original_n_id
        edge_index = torch.empty(size=(2,0)).long()
        e_id = neighbor_loader.e_id[n_id]
        for _ in range(model.num_gnn_layers):
            n_id, edge_index, e_id = neighbor_loader(n_id)

        helper[n_id] = torch.arange(n_id.size(0), device=device)
        
        # Model forward
        # NOTE: src_emb, pos_dst_emb are the embedding that will be saved in memory
        pos_out, neg_out, src_emb, pos_dst_emb = model(batch=batch, n_id=n_id, msg=data.msg[e_id].to(device),
                                                       t=data.t[e_id].to(device), edge_index=edge_index, id_mapper=helper)

        if neg_sampler is None:
            y_true = batch.y.cpu()
            y_pred = pos_out.detach().cpu()
            y_pred_list.append(torch.argmax(y_pred, -1) if multiclass else y_pred)
        else:
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))

            y_pred = torch.cat([pos_out, neg_out], dim=0).cpu()
            y_true = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))],
                               dim=0)    
            y_pred_list.append((y_pred.sigmoid() > 0.5).float())

        y_pred_confidence_list.append(y_pred)
        y_true_list.append(y_true)

        # Update memory and neighbor loader with ground-truth state.
        model.update(src, pos_dst, t, msg, src_emb, pos_dst_emb)
        neighbor_loader.insert(src, pos_dst)

    t1 = time.time()

    # Compute scores  
    y_true_list = torch.cat(y_true_list)
    if neg_sampler is not None: y_true_list = y_true_list.unsqueeze(1)
    y_pred_list = torch.cat(y_pred_list)
    y_pred_confidence_list = torch.cat(y_pred_confidence_list)
    scores = scoring(y_true_list, y_pred_list, y_pred_confidence_list, 
                     is_regression=regression, is_multiclass=multiclass)
    scores['loss'] = criterion(y_pred_confidence_list, y_true_list).item() 
    scores['time'] = datetime.timedelta(seconds=t1 - t0)

    true_values = (y_true_list, y_pred_list, y_pred_confidence_list) if return_predictions else None
    if wandb_log:
        for k, v in scores.items():
            if  k == 'confusion_matrix':
                continue
            else:
                wandb.log({f"{eval_name} {k}, {neg_sampler}":v if k != 'time' else v.total_seconds()}, commit=False)
                
        _cm = wandb.plot.confusion_matrix(preds=y_pred_list.squeeze().numpy(),
                                          y_true=y_true_list.squeeze().numpy(),
                                          class_names=["negative", "positive"])
        wandb.log({f"conf_mat {eval_name}, {neg_sampler}" : _cm}, commit='val' in eval_name or 'test' in eval_name)
        
    return scores, true_values


@ray.remote(num_cpus=int(os.environ.get('NUM_CPUS_PER_TASK', 1)), num_gpus=float(os.environ.get('NUM_GPUS_PER_TASK', 0.)))
def link_prediction(model_instance, conf):
    return link_prediction_single(model_instance, conf)


@ray.remote(num_cpus=int(os.environ.get('NUM_CPUS_PER_TASK', 4)), num_gpus=float(os.environ.get('NUM_GPUS_PER_TASK', 0.)))
def tgb_link_prediction(model_instance, conf):
    return tgb_link_prediction_single(model_instance, conf)


def tgb_link_prediction_single(model_instance, conf):
    print(conf)
    # Set the configuration seed
    set_seed(conf['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data loading
    data, _, _, _, _, _, dataset = get_dataset(root=conf['data_dir'], name=conf['data_name'], seed=conf['exp_seed'])
    # dataset = PyGLinkPropPredDataset(name=conf['data_name'], root=conf['data_dir'])
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask

    data = data.to(device)
    data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
    metric = dataset.eval_metric

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    train_loader = TemporalDataLoader(train_data, batch_size=conf["batch"])
    val_loader = TemporalDataLoader(val_data, batch_size=conf["batch"])
    test_loader = TemporalDataLoader(test_data, batch_size=conf["batch"])

    # neighhorhood sampler
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=conf['sampler']['size'], device=device)

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    # Define model
    model = model_instance(**conf['model_params']).to(device)

    conf['model_size'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=conf['optim_params']['lr'], 
                                 weight_decay=conf['optim_params']['wd'])
    criterion = torch.nn.BCEWithLogitsLoss()
 
    lr_scheduler = FakeScheduler()
    if conf["lr_scheduler"]:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=conf["lr_scheduler_patience"], mode='max')

    evaluator = Evaluator(name=conf['data_name'])
    neg_sampler = dataset.negative_sampler

    history = []
    best_epoch = 1
    best_score = -np.inf
 
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
    dataset.load_val_ns()
    val_perf_list = []
    try: 
        for e in range(best_epoch, conf['epochs']):
            t0 = time.time()

            train_losses: list[float] = tgb_train(
                data=data,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                train_loader=train_loader,
                criterion=criterion,
                neighbor_loader=neighbor_loader,
                helper=assoc, 
                device=device,
                pbar=conf["debug"]
            )

            print('\nEpoch {:d}. avg loss {:.5f}\n'.format(e, np.array(train_losses).mean()))

            vl_scores = {"mrr": -np.inf}
            tr_scores = {'loss': np.inf}

            if e == 1 or (e % conf["validate_every"] == 0 or e >= conf['epochs']):
                tr_scores = {'loss': float(np.array(train_losses).mean())}

                print(f'Running tgb_test bc {e=} or {conf["validate_every"]=}')

                model.reset_memory()
                neighbor_loader.reset_state()

                _, _ = tgb_test(
                    data=data, 
                    model=model,
                    loader=train_loader,
                    neg_sampler=neg_sampler,
                    neighbor_loader=neighbor_loader,
                    split_mode="train",
                    helper=assoc,
                    evaluator=evaluator,
                    metric=metric,
                    device=device,
                    pbar=conf['debug']
                )
                
                # validation
                # dict[str, float]
                vl_scores, _ = tgb_test(
                    data=data, 
                    model=model,
                    loader=val_loader,
                    neg_sampler=neg_sampler,
                    neighbor_loader=neighbor_loader,
                    split_mode="val_fast",
                    helper=assoc,
                    evaluator=evaluator,
                    metric=metric,
                    device=device,
                    validation_subsample=conf["validation_subsample"],
                    batched_val=conf["validation_batched"],
                    pbar=conf["debug"]
                )

                print("End of tgb_test:")
                print(f'Train :{tr_scores}')
                print(f'Val :{vl_scores}\n')

                val_perf_list.append(vl_scores[conf['metric']])
        

            lr_scheduler.step(vl_scores[conf['metric']]) 
                
            history.append({
                'train': tr_scores,
                'val': vl_scores
            })

            if len(history) == 1 or vl_scores[conf['metric']] > best_score:
                best_score = vl_scores[conf['metric']]
                best_epoch = e
                print(f"\nSaving Best model at epoch {e}\n")
                torch.save({
                    'train_ended': False,
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'best_score': best_score,
                    'loss': (tr_scores["loss"], None, None),
                    'tr_scores': tr_scores,
                    'vl_scores': vl_scores,
                    'true_values': (None, None, None),
                    'history': history
                }, path_save_best)
                
            epoch_times.append(time.time()-t0)

            print(f'Epoch {e}: {np.mean(epoch_times)} +/- {np.std(epoch_times)} seconds per epoch') 

            if e - best_epoch > conf['patience']:
                break
            if np.isnan(tr_scores["loss"]):
                break
            if vl_scores['mrr'] == 1.0:
                break


        # Evaluate on test
        print('Evaluating from scratch. ')
        print(f'Loading model at epoch {best_epoch:d}...')
        
        ckpt = torch.load(path_save_best, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        model.reset_memory()
        neighbor_loader.reset_state()

        tr_scores, _ = tgb_test(
            data=data, 
            model=model,
            loader=train_loader,
            neg_sampler=neg_sampler,
            neighbor_loader=neighbor_loader,
            split_mode="train",
            helper=assoc,
            evaluator=evaluator,
            metric=metric,
            device=device,
            pbar=conf['debug']
        )

        vl_scores, _ = tgb_test(
            data=data, 
            model=model,
            loader=val_loader,
            neg_sampler=neg_sampler,
            neighbor_loader=neighbor_loader,
            split_mode="val",
            helper=assoc,
            evaluator=evaluator,
            metric=metric,
            device=device,
            pbar=conf['debug']
        )

        dataset.load_test_ns()

        ts_scores, _ = tgb_test(
            data=data, 
            model=model,
            loader=test_loader,
            neg_sampler=neg_sampler,
            neighbor_loader=neighbor_loader,
            split_mode="test",
            helper=assoc,
            evaluator=evaluator,
            metric=metric,
            device=device,
            pbar=conf['debug']
        )

        ckpt['test_score'] = ts_scores
        ckpt['val_score'] = vl_scores
        ckpt['train_score'] = tr_scores
        ckpt['train_ended'] = True
        torch.save(ckpt, path_save_best)

        history = ckpt['history'] if conf['log'] else None
    except RuntimeError as e:
        # Allow to fail for gradient explosion
        # Make a moot results run
        ckpt = {}
        ckpt['test_score'] = {conf['metric']: -np.inf}
        ckpt['val_score'] = {conf['metric']: -np.inf}
        ckpt['train_score'] = {conf['metric']: -np.inf}
        ckpt['train_ended'] = True
        ckpt["epoch"] = -1
        history = None

    return ckpt['test_score'], ckpt['val_score'], ckpt['train_score'], ckpt['epoch'], conf, history


def link_prediction_single(model_instance, conf):
    if conf['wandb']:
        wandb.init(project=conf['data_name'], group=conf['model'], config=conf)

    # Set the configuration seed
    set_seed(conf['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data, _, _, _, _, _, _ = get_dataset(root=conf['data_dir'], 
                                      name=conf['data_name'], 
                                      seed=conf['exp_seed'])

    train_data, val_data, test_data = data.train_val_test_split(val_ratio=conf['split'][0], test_ratio=conf['split'][1])
    train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)

    train_loader = TemporalDataLoader(train_data, batch_size=conf['batch'])
    val_loader = TemporalDataLoader(val_data, batch_size=conf['batch'])
    test_loader = TemporalDataLoader(test_data, batch_size=conf['batch'])

    neighbor_loader = LastNeighborLoader(data.num_nodes, size=conf['sampler']['size'], device=device)
    
    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    # Define model
    if conf['model'] == edgebank:
        conf['model_params']['timespan'] = train_data.t.max() - train_data.t.min()
    model = model_instance(**conf['model_params']).to(device)

    if conf['regression']:
        criterion = REGRESSION_SCORES[conf['metric']]
    elif conf['multiclass']:
        criterion = torch.nn.CrossEntropyLoss()
    else: 
        criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=conf['optim_params']['lr'], 
                                 weight_decay=conf['optim_params']['wd'])
    
    lr_scheduler = FakeScheduler()
    if conf['lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=conf["lr_scheduler_patience"], mode='min' if conf['regression'] else 'max')
    
    (train_src_nodes, train_dst_nodes, 
     val_src_nodes, val_dst_nodes, 
     test_src_nodes,test_dst_nodes) = get_node_sets(conf['strategy'], train_data, val_data, test_data)

    if conf['regression'] or conf['multiclass']:
        train_neg_link_sampler = None
        val_neg_link_sampler = None
        test_neg_link_sampler = None
    else:
        neg_sampler_instance = getattr(negative_sampler, conf['neg_sampler'])
        train_neg_link_sampler = neg_sampler_instance(train_src_nodes, train_dst_nodes, name='train', 
                                                      check_link_existence=not conf['no_check_link_existence'],
                                                      seed=conf['exp_seed']+1)
        val_neg_link_sampler = neg_sampler_instance(val_src_nodes, val_dst_nodes, name='val', 
                                                    check_link_existence=not conf['no_check_link_existence'],
                                                    seed=conf['exp_seed']+2)
        test_neg_link_sampler = neg_sampler_instance(test_src_nodes, test_dst_nodes, name='test', 
                                                     check_link_existence=not conf['no_check_link_existence'],
                                                     seed=conf['exp_seed']+3)

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

        pbar = False
        if conf['debug']:
            print('Epoch {:d}:'.format(e))
            pbar = True

        train_losses: List[float] = train(
            data=data,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            criterion=criterion,
            neighbor_loader=neighbor_loader,
            train_neg_sampler=train_neg_link_sampler,
            helper=assoc, 
            requires_grad=conf['model']!=edgebank,
            device=device,
            pbar=pbar
        )
        
        model.reset_memory()
        neighbor_loader.reset_state()

        tr_scores, _ = eval(data=data, model=model, loader=train_loader, criterion=criterion, 
                            neighbor_loader=neighbor_loader, neg_sampler=train_neg_link_sampler, 
                            regression=conf['regression'], multiclass=conf['multiclass'], helper=assoc, 
                            eval_seed=conf['exp_seed'], device=device, eval_name='train_eval',
                            wandb_log=conf['wandb'], pbar=pbar)
        
        if conf['reset_memory_eval']:
            model.reset_memory()

        vl_scores, vl_true_values = eval(data=data, model=model, loader=val_loader, criterion=criterion, 
                                        neighbor_loader=neighbor_loader, neg_sampler=val_neg_link_sampler, 
                                        regression=conf['regression'], multiclass=conf['multiclass'],
                                        helper=assoc, eval_seed=conf['exp_seed'], device=device,
                                        eval_name='val_eval', wandb_log=conf['wandb'], pbar=pbar)
        
        lr_scheduler.step(vl_scores[conf['metric']]) # only for pascal voc

        history.append({
            'train': tr_scores,
            'val': vl_scores
        })

        if len(history) == 1 or isbest(vl_scores[conf['metric']], best_score, conf['regression']):
            best_score = vl_scores[conf['metric']]
            best_epoch = e
            torch.save({
                'train_ended': False,
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(), # only for pascal voc
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

    if conf['use_all_strategies_eval']:
        strategies = dst_strategies
    else:
        strategies = [conf['strategy']]

    for strategy in strategies:
        if conf['regression'] or conf['multiclass']:
            tmp_train_neg_link_sampler = None
            tmp_val_neg_link_sampler = None
            tmp_test_neg_link_sampler = None
        elif strategies == conf['strategy']:
            tmp_train_neg_link_sampler = train_neg_link_sampler
            tmp_val_neg_link_sampler = val_neg_link_sampler
            tmp_test_neg_link_sampler = test_neg_link_sampler
        else:
            (tmp_train_src_nodes, tmp_train_dst_nodes, 
             tmp_val_src_nodes, tmp_val_dst_nodes, 
             tmp_test_src_nodes, tmp_test_dst_nodes) = get_node_sets(strategy, train_data, val_data, test_data)

            neg_sampler_instance = getattr(negative_sampler, conf['neg_sampler'])
            tmp_train_neg_link_sampler = neg_sampler_instance(tmp_train_src_nodes, tmp_train_dst_nodes,
                                                              check_link_existence=not conf['no_check_link_existence'],
                                                              name='train', seed=conf['exp_seed']+1)
            tmp_val_neg_link_sampler = neg_sampler_instance(tmp_val_src_nodes, tmp_val_dst_nodes,
                                                            check_link_existence=not conf['no_check_link_existence'],
                                                            name='val', seed=conf['exp_seed']+2)
            tmp_test_neg_link_sampler = neg_sampler_instance(tmp_test_src_nodes, tmp_test_dst_nodes,
                                                             check_link_existence=not conf['no_check_link_existence'],
                                                             name='test', seed=conf['exp_seed']+3)

        model.reset_memory()
        neighbor_loader.reset_state()

        tr_scores, tr_true_values = eval(data=data, model=model, loader=train_loader, criterion=criterion, 
                                         neighbor_loader=neighbor_loader, neg_sampler=tmp_train_neg_link_sampler, 
                                         regression=conf['regression'], multiclass=conf['multiclass'],
                                         helper=assoc, eval_seed=conf['exp_seed'], device=device, 
                                         eval_name='train_eval', wandb_log=conf['wandb'])
        
        if conf['reset_memory_eval']:
            model.reset_memory()

        vl_scores, vl_true_values = eval(data=data, model=model, loader=val_loader, criterion=criterion, 
                                         neighbor_loader=neighbor_loader, neg_sampler=tmp_val_neg_link_sampler, 
                                         regression=conf['regression'], multiclass=conf['multiclass'],
                                         helper=assoc, eval_seed=conf['exp_seed'], device=device, 
                                         eval_name='val_eval', wandb_log=conf['wandb'])
        
        if conf['reset_memory_eval']:
            model.reset_memory()

        ts_scores, ts_true_values = eval(data=data, model=model, loader=test_loader, criterion=criterion, 
                                         neighbor_loader=neighbor_loader, neg_sampler=tmp_test_neg_link_sampler, 
                                         regression=conf['regression'], multiclass=conf['multiclass'],
                                         helper=assoc, eval_seed=conf['exp_seed'], device=device, 
                                         eval_name='test_eval', wandb_log=conf['wandb'])

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
