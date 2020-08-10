import os
import copy
import glob
import shutil
import pandas as pd
import numpy as np
import random

import torch
from torch_scatter import scatter_add
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip


class MCDataset(InMemoryDataset):
    def __init__(self, root, name, num_neg=2, transform=None, pre_transform=None):
        """
        Matrix Completion dataset.

        Arguments:
        ----------
        root : `str'
            Path to directory where raw & processed files can be found.
        name : `str'
            Dataset name
        num_neg : `int'
            Number of negative samples to sample per positive sample in loss function. (Default=2)
        overide_items : `int'
            Number of item nodes to create if require nodes with no edges or data.
        """
        self.name = name
        self.num_neg = num_neg
        super(MCDataset, self).__init__(root, transform, pre_transform)
        # processed_path[0] is the processed data, defined by process method
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def num_nodes(self):
        """Number of nodes in graph."""
        return self.data.x.shape[0]

    @property
    def raw_file_names(self):
        """File name in root directory for raw data."""
        return ['train_triplet_df.pkl']

    @property
    def processed_file_names(self):
        """File name of processed files."""
        return 'data.pt'

    def process(self):
        """Process raw data files into PyTorch Geometric `Data' objects"""
        # select dataset
        if self.name == 'bets':
            path = self.raw_paths[0]
        else:
            raise ValueError()
        
        # create dataframe and get glocal counts
        train_df, train_nums = self.create_df(path)
        
        # create ids for edge dge
        train_idx, train_gt = self.create_gt_idx(train_df, train_nums)

        # add constant so no userId and itemID are equal (nodes have unique ids)
        train_df['itemId'] = train_df['itemId'] + train_nums['user']
        
        # node features simply index (or one-hot)
        x = torch.arange(train_nums['node'], dtype=torch.long).unsqueeze(dim=1)

        # Prepare edges
        edge_user = torch.tensor(train_df['userId'].values.astype('int64'))
        edge_item = torch.tensor(train_df['itemId'].values.astype('int64'))

        # create edge index of shape 2 x (2*n_edges), counted twice since undirected 
        edge_index = torch.stack((torch.cat((edge_user, edge_item), 0),
                                  torch.cat((edge_item, edge_user), 0)), 0)
        edge_index = edge_index.to(torch.long) # correct data type

        # prepare edge values
        edge_type = torch.tensor(train_df['rating'].values)
        edge_type = torch.cat((edge_type, edge_type), 0)

        # create array for normalisation constants
        edge_norm = copy.deepcopy(edge_index[1])

        # calculate normalisation constant per node
        for idx in range(train_nums['node']):
            # sum in-degree, all weights going into node
            if idx in train_df['userId'].values:
                count = train_df[train_df['userId']==idx]['rating'].values.sum()
            elif idx in train_df['itemId'].values:
                count = train_df[train_df['itemId']==idx]['rating'].values.sum()
            else:
                count = 0
            # store in relevant location in edge_norm
            edge_norm = torch.where(edge_norm==idx,
                                    torch.tensor(count),
                                    edge_norm)
        # take inverse for convenience
        edge_norm = edge_norm.to(torch.float)
        edge_norm = torch.where(edge_norm>0,1/edge_norm,torch.tensor(0.0))
        
        # prepare Data object
        data = Data(x=x, edge_index=edge_index)
        data.edge_type = edge_type
        data.edge_norm = edge_norm        
        data.num_users = torch.tensor([train_nums['user']])
        data.num_items = torch.tensor([train_nums['item']])
        
        # get correct negative indices for loss function
        if self.num_neg>0:
            # total number of negative items to sample
            n_neg = int(self.num_neg*train_nums['edge'])
            # list of all potential valid negative ids corresponding to user-item pairs without an interaction
            full_neg_idx = list(set([i for i in range(train_nums['user']*train_nums['item'])])-set(train_idx.tolist()))
            # select n_neg ids from full_neg_idx
            neg_idx = torch.Tensor(random.sample(full_neg_idx, n_neg)).long()
            # shuffle and store values
            perm = torch.randperm((self.num_neg+1)*train_nums['edge']).long()
            data.train_idx = torch.cat((train_idx, neg_idx),0)[perm] # ids
            data.train_gt = torch.cat((train_gt, torch.ones(n_neg).type(torch.double)),0)[perm] # count value
            data.train_rt = torch.cat((torch.ones(train_nums['edge']), torch.zeros(n_neg)),0)[perm] # indicator of observed or not

            del n_neg, full_neg_idx, neg_idx, perm
        else:
            data.train_idx = train_idx
            data.train_gt = train_gt
            data.train_rt = torch.ones(train_nums['edge'])

        # save data object
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def create_df(self, pkl_path):
        """
        Takes raw pickle file path and returns processed dataframe and global counts.
        
        Parameters:
        ----------        
        pkl_path : `str`
            Path to raw pickle file of data.

        Returns:
        --------
        df : `pandas.DataFrame`
            DataFrame with user-item interactions.

        nums : `dict'
            Dictionary with number of users, items, nodes and edges.
        """
        df = pd.read_pickle(pkl_path)
        n_users = int(df['userId'].nunique())
        n_items = int(df['itemId'].nunique()) if not self.overide_items else int(self.overide_items)
        nums = {'user': n_users,
                'item': n_items,
                'node': n_users + n_items,
                'edge': len(df)}
        
        return df, nums
    
    def create_gt_idx(self, df, nums):
        """Creates unique id per edge. Returns ids and edge values as Tensors."""
        df['idx'] = df['userId'] * nums['item'] + df['itemId']
        idx = torch.tensor(df['idx'].values.astype('int64'))
        gt = torch.tensor(df['rating'].values.astype('float64'))
        return idx, gt

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data[0]

    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)

class MCBatchDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        """
        Matrix Completion dataset for batched graph.

        Arguments:
        ----------
        root : `str'
            Path to directory where raw & processed files can be found.
        name : `str'
            Dataset name
        """
        self.name = name
        super(MCBatchDataset, self).__init__(root, transform, pre_transform)
        # processed_path[0] is the processed data, defined by process method
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_nodes(self):
        """Number of nodes in graph."""
        return self.data.x.shape[0]

    @property
    def raw_file_names(self):
        """File name in root directory for raw data."""
        return ['train_triplet_df.pkl']

    @property
    def processed_file_names(self):
        """File name of processed files."""
        return 'data.pt'

    def process(self):
        """Process raw data files into PyTorch Geometric `Data' objects"""
        # select dataset
        if self.name == 'bets':
            path = self.raw_paths[0]
        else:
            raise ValueError()

        # create dataframe and get glocal counts
        train_df, train_nums = self.create_df(path)

        # add constant so no userId and itemID are equal (nodes have unique ids)
        train_df['itemId'] = train_df['itemId'] + train_nums['user']
        
        # node features simply index (or one-hot)
        x = torch.arange(train_nums['node'], dtype=torch.long).unsqueeze(dim=1)

        # Prepare edges
        edge_user = torch.tensor(train_df['userId'].values.astype('int64'))
        edge_item = torch.tensor(train_df['itemId'].values.astype('int64'))

        edge_index = torch.cat((edge_user, edge_item), 0)
        edge_index = edge_index.to(torch.long)

        # prepare edge values
        edge_type = torch.tensor(train_df['rating'].values)
        
        # create array for normalisation constants
        edge_norm = torch.zeros(train_nums['node'], dtype=torch.float)
        # calculate normalisation constant per node 
        for idx in range(train_nums['node']):
            if idx%10000==0:
                print('Processing node {}'.format(idx))
            # sum in-degree, all weights going into node
            if idx in train_df['userId'].values:
                count = train_df[train_df['userId']==idx]['rating'].values.sum()
            elif idx in train_df['itemId'].values:
                count = train_df[train_df['itemId']==idx]['rating'].values.sum()
            else:
                count = 0
            # store in relevant location in edge_norm
            edge_norm[idx] = torch.tensor(count)

        # take inverse for convenience
        edge_norm = edge_norm.to(torch.float)
        edge_norm = torch.where(edge_norm>0,1/edge_norm,torch.tensor(0.0))
        
        # Prepare Data object
        data = Data(x=x, edge_index=edge_index)
        data.edge_user = edge_user
        data.edge_item = edge_item
        data.edge_type = edge_type
        data.edge_norm = edge_norm        
        data.num_users = torch.tensor([train_nums['user']])
        data.num_items = torch.tensor([train_nums['item']])
        data.num_nodes = torch.tensor([train_nums['node']])
        
        # save data object                         
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def create_df(self, pkl_path):
        """
        Takes raw pickle file path and returns processed dataframe and global counts.
        
        Parameters:
        ----------        
        pkl_path : `str`
            Path to raw pickle file of data.

        Returns:
        --------
        df : `pandas.DataFrame`
            DataFrame with user-item interactions.

        nums : `dict'
            Dictionary with number of users, items, nodes and edges.
        """
        df = pd.read_pickle(pkl_path)
        n_users = int(df['userId'].nunique())
        n_items = int(df['itemId'].nunique()) if not self.overide_items else int(self.overide_items)
        nums = {'user': n_users,
                'item': n_items,
                'node': n_users + n_items,
                'edge': len(df)}
        
        return df, nums
    
    def create_gt_idx(self, df, nums):
        """Creates unique id per edge. Returns ids and edge values as Tensors."""
        df['idx'] = df['userId'] * nums['item'] + df['itemId']
        idx = torch.tensor(df['idx'].values.astype('int64'))
        gt = torch.tensor(df['rating'].values.astype('float64'))
        return idx, gt

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data[0]

    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MCDataset(root='data/lastfm/', name='lastfm')
    data = dataset[0]
    print(data)
    data = data.to(device)