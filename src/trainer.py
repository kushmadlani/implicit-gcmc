import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import subgraph
import wandb

import os
os.environ['WANDB_API_KEY'] = 'f2935f67cdf03fb8a19d09e8bc7124024891dc3f'

from memory_profiler import profile

class Batcher(Dataset):
    def __init__(self, n_edge):
        self.n_edge = n_edge
        
    def __getitem__(self, index):
        return index
    
    def __len__(self):
        return self.n_edge 


class Trainer:
    def __init__(self, model, data, optimizer, cfg,
                 calc_eval=False, calc_metrics=None, top_n=None, test_data=None, batch_ratio=None, batch_neg=None):
        self.model = model
        # self.dataset = dataset
        self.data = data
        self.optimizer = optimizer
        self.loss_fn = torch.nn.BCELoss()

        # counts
        self.n_users = cfg.num_users
        self.n_items = cfg.num_nodes - self.n_users

        # batching
        self.batch = True if batch_ratio>0 else False
        if batch_ratio:
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
            self.test_mat = test_data['val']
            self.test_masked_mat = test_data['val_masked']

            

    def training(self, epochs):
        best_mpr = 1
        self.epochs = epochs
        for epoch in range(self.epochs):
            if self.batch:
                sub_graphs, loss = self.train_one_batch()
            else:
                loss = self.train_one()

            if self.calc_eval:
                if self.batch:
                    MAP, rec_at_k, mpr_all, mpr_mask = self.test_batch(sub_graphs)
                else:
                    MAP, rec_at_k, mpr_all, mpr_mask = self.test()

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

    def train_one(self):
        self.data = self.data.to(self.device)
        self.model.train()
        epoch_loss = 0

        out = self.model(self.data.x, self.data.edge_index, self.data.edge_norm)
        # loss_fn = torch.nn.BCEWithLogitsLoss(weight=self.data.train_gt)
        out = out.squeeze(dim=1)
        # self.testnan = torch.isnan(out.reshape(self.data.num_users, self.data.num_items))
        loss = self.loss_fn(out[self.data.train_idx.long()].float(), self.data.train_rt.float()).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        epoch_loss = loss.item()

        return epoch_loss

    @profile
    def train_one_batch(self):
        self.model.train()
        epoch_loss = 0
        sub_graphs = []

        for batch in self.dataloader:
            sub_data = subgraph(batch, self.data, self.n_neg)
            sub_data = sub_data.to(self.device)
            print(sub_data)
            sub_graphs.append(sub_data)
            # print(sub_data.num_users, sub_data.num_items, sub_data.num_users*sub_data.num_items)
            # print(sub_data.num_users, sub_data.num_items)
            out = self.model(sub_data.x, sub_data.edge_index, sub_data.edge_norm, sub_data.users)
            out = out.squeeze(dim=1)
            # print(out.size())
            loss = self.loss_fn(out[sub_data.train_idx.long()].float(), sub_data.train_rt.float()).to(self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('batch loss in {}'.format(loss.item()))
            epoch_loss += loss.item()

        return sub_graphs, epoch_loss


    def test(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index, self.data.edge_norm)
            out = out.reshape(self.data.num_users, self.data.num_items).numpy()
            MAP, rec_at_k, mpr_all, mpr_mask = self.calc_metrics(out, self.test_mat, self.test_masked_mat, self.top_n)
        return MAP, rec_at_k, mpr_all, mpr_mask

    @profile
    def test_batch(self, list_sub_graphs):
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
            # print('stitched matrix shape is {}'.format(out.shape))
            MAP, rec_at_k, mpr_all, mpr_mask = self.calc_metrics(out, self.test_mat, self.test_masked_mat, self.top_n)
        return MAP, rec_at_k, mpr_all, mpr_mask

    def summary(self, epoch, loss, MAP=None, rec_at_k=None, mpr_all=None, mpr_mask=None):
        if MAP is None:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} ]'.format(
                epoch, self.epochs, loss))
        else:
            print('[ Epoch: {:>4}/{} | Loss: {:.6f} | MAP: {:.6f} | Recall@k: {:.6f} | MPR: {:.6f} | MPR_masked: {:.6f} ]'.format(
                epoch, self.epochs, loss, MAP, rec_at_k, mpr_all, mpr_mask))
