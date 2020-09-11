import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import subgraph
import wandb

import os
os.environ['WANDB_API_KEY'] = 'f2935f67cdf03fb8a19d09e8bc7124024891dc3f'

from memory_profiler import profile


def save_checkpoint(state, ep, filename='checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    print("=> Saving a new best at epoch:", ep)
    torch.save(state, filename)  # save checkpoint

class Trainer:
    def __init__(self, model, data, optimizer, cfg, calc_eval=False, calc_metrics=None, top_n=None, val_data=None):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.loss_fn = torch.nn.BCELoss()

        # counts
        self.n_users = cfg.num_users
        self.n_items = cfg.num_nodes - self.n_users
        self.use_gpu = cfg.use_gpu
        self.device = cfg.device

        # testing params and data
        self.calc_eval = calc_eval
        if self.calc_eval:
            self.calc_metrics = calc_metrics
            self.top_n = top_n
            self.val_mat = val_data['val']
            self.val_masked_mat = val_data['val_masked']
        
        self.model_name = 'checkpoint_'+str(cfg.dataset)+'_'+str(cfg.f)+'_'+str(cfg.item_side_info)+'_.pth.tar'

    def training(self, epochs):
        best_mpr = 1
        self.epochs = epochs

        for epoch in range(self.epochs):
            loss = self.train_one()                
            
            if self.calc_eval:
                MAP, rec_at_k, mpr_all, mpr_mask = self.test(self.val_mat , self.val_masked_mat)
                self.summary(epoch, loss, MAP, rec_at_k, mpr_all, mpr_mask)
                wandb.log({
                    'Training Loss': loss,
                    'MAP@N': MAP,
                    'Recall@N': rec_at_k,
                    'MPR (all)': mpr_all,
                    'MPR (new)': mpr_mask
                })
                if mpr_mask<best_mpr:
                    wandb.run.summary["best_mpr"] = mpr_mask
                    best_mpr = mpr_mask
            else:
                self.summary(epoch, loss)

        print('END TRAINING')
        torch.save(self.model.state_dict(), self.model_name)
        print("Model saved")

    def train_one(self):
        self.data = self.data.to(self.device)
        self.model.train()
        epoch_loss = 0

        out = self.model(self.data.x, self.data.edge_index, self.data.edge_norm)
        out = out.squeeze(dim=1)
        loss = self.loss_fn(out[self.data.train_idx.long()].float(), self.data.train_rt.float()).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        epoch_loss = loss.item()

        return epoch_loss

    def test(self, mat, masked_mat):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_norm)
            out = out.reshape(self.data.num_users, self.data.num_items).numpy()
            MAP, rec_at_k, mpr_all, mpr_mask = self.calc_metrics(out, mat, masked_mat, self.top_n)
        return MAP, rec_at_k, mpr_all, mpr_mask

    def summary(self, epoch, loss, MAP=None, rec_at_k=None, mpr_all=None, mpr_mask=None):
        if MAP is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch+1, self.epochs, loss))
        else:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | MAP: {:.6f} | Recall@k: {:.6f} | MPR: {:.6f} | MPR_masked: {:.6f} ]'.format(
                epoch+1, self.epochs, loss, MAP, rec_at_k, mpr_all, mpr_mask))

    
class Batcher(Dataset):
    def __init__(self, n_edge):
        self.n_edge = n_edge
        
    def __getitem__(self, index):
        return index
    
    def __len__(self):
        return self.n_edge 

class TrainerBatch:
    def __init__(self, model, data, optimizer, cfg,
                 calc_eval=False, calc_metrics=None, top_n=None, val_data=None, batch_ratio=None, batch_neg=None):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.loss_fn = torch.nn.BCELoss()

        # counts
        self.n_users = cfg.num_users
        self.n_items = cfg.num_nodes - self.n_users

        # batching
        self.batch = True 
        self.batch_ratio = batch_ratio
        num_user = int(data.num_users)
        batch_size = 1+num_user//self.batch_ratio
        batcher = Batcher(n_edge=num_user)
        self.dataloader = DataLoader(batcher, batch_size=batch_size, shuffle=True)
        self.n_neg = batch_neg
        
        self.use_gpu = cfg.use_gpu
        self.device = cfg.device

        # testing params and data
        self.calc_eval = calc_eval
        if self.calc_eval:
            self.calc_metrics = calc_metrics
            self.top_n = top_n
            self.val_mat = val_data['val']
            self.val_masked_mat = val_data['val_masked']
        
        self.model_name = 'checkpoint_'+str(cfg.dataset)+'_'+str(cfg.f)+'_'+str(cfg.item_side_info)+'_.pth.tar'

    def training(self, epochs, n_rebatch):
        best_mpr = 1
        self.epochs = epochs

        sub_graphs = []
        for batch in self.dataloader:
            sub_data = subgraph(batch, self.data, self.n_neg)
            sub_data = sub_data.to(self.device)
            print(sub_data)
            sub_graphs.append(sub_data)

        for epoch in range(self.epochs):
        
            if (epoch+1) % n_rebatch == 0:
                print('new graphs')
                sub_graphs = []
                for batch in self.dataloader:
                    sub_data = subgraph(batch, self.data, self.n_neg)
                    sub_data = sub_data.to(self.device)
                    print(sub_data)
                    sub_graphs.append(sub_data)

            loss = self.train_one_batch(sub_graphs)

            if self.calc_eval:
                MAP, rec_at_k, mpr_all, mpr_mask = self.test(self.val_mat , self.val_masked_mat)
                self.summary(epoch, loss, MAP, rec_at_k, mpr_all, mpr_mask)
                wandb.log({
                    'Training Loss': loss,
                    'MAP@N': MAP,
                    'Recall@N': rec_at_k,
                    'MPR (all)': mpr_all,
                    'MPR (new)': mpr_mask
                })
                if mpr_mask<best_mpr:
                    wandb.run.summary["best_mpr"] = mpr_mask
                    best_mpr = mpr_mask
            else:
                self.summary(epoch, loss)

        print('END TRAINING')


    def train_one_batch(self, sub_graphs):
        self.model.train()
        epoch_loss = 0

        for graph in sub_graphs:
            out = self.model(graph.x, graph.edge_index, graph.edge_norm, graph.users)
            out = out.squeeze(dim=1)
            loss = self.loss_fn(out[graph.train_idx.long()].float(), graph.train_rt.float()).to(self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('batch loss in {}'.format(loss.item()))
            epoch_loss += loss.item()

        return epoch_loss

    def test(self, list_sub_graphs, mat, masked_mat):
        self.model.eval()
        with torch.no_grad():
            users, values = [], []
            for graph in list_sub_graphs:
                sub_data = graph
                out = self.model(sub_data.x, sub_data.edge_index, sub_data.edge_norm, sub_data.users)
                if self.use_gpu:
                    out = out.reshape(int(sub_data.num_users), self.data.num_items).cpu().numpy()
                else:
                    out = out.reshape(int(sub_data.num_users), self.data.num_items).numpy()
                # stitching
                users.extend(sub_data.users.tolist())
                values.append(out)
            out = np.concatenate(values,axis=0)
            out = out[users]
            MAP, rec_at_k, mpr_all, mpr_mask = self.calc_metrics(out, mat, masked_mat, self.top_n)
        return MAP, rec_at_k, mpr_all, mpr_mask

    def summary(self, epoch, loss, MAP=None, rec_at_k=None, mpr_all=None, mpr_mask=None):
        if MAP is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch+1, self.epochs, loss))
        else:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | MAP: {:.6f} | Recall@k: {:.6f} | MPR: {:.6f} | MPR_masked: {:.6f} ]'.format(
                epoch+1, self.epochs, loss, MAP, rec_at_k, mpr_all, mpr_mask))

