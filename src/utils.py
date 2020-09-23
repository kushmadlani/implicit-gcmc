import time
import math
import numpy as np
import math
import random
from scipy import stats
from time import perf_counter 

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


def ster_uniform(tensor, in_dim, out_dim):
    if tensor is not None:
        tensor.data.uniform_(-0.001, 0.001)

def random_init(tensor, in_dim, out_dim):
    thresh = math.sqrt(6.0 / (in_dim + out_dim))
    if tensor is not None:
        try:
            tensor.data.uniform_(-thresh, thresh)
        except:
            nn.init.uniform_(tensor, a=-thresh, b=thresh)

def init_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            truncated_normal(m.bias)
        except:
            pass

def init_uniform(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight)
        try:
            truncated_normal(m.bias)
        except:
            pass

def truncated_normal(tensor, mean=0, std=1):
    tensor.data.fill_(std * 2)
    with torch.no_grad():
        while(True):
            if tensor.max() >= std * 2:
                tensor[tensor>=std * 2] = tensor[tensor>=std * 2].normal_(mean, std)
                tensor.abs_()
            else:
                break


def m_p_r(R_test, R_hat, verbose=False):
    """(n_users, n_items) returns percentage ranking from two dense matrices"""
    if sp.issparse(R_test):
        R_true = np.array(R_test.todense())
    else:
        R_true = R_test

    (n, m) = R_hat.shape
    R_r = np.zeros(shape=(n,m))

    for i in range(n):
        R_r[i,:] = stats.rankdata(-R_hat[i], "average")/m
        if i%10000==0 and verbose:
            print('processing user {}'.format(i))
    
    r = np.einsum('ij,ij',R_true,R_r)/np.sum(R_true)

    return r

def sparse_to_list(X):
    """list of nonzero indices per row of sparse X"""
    result = np.split(X.indices, X.indptr)[1:-1]
    result = [list(r) for r in result]
    return result

def apk(actual, predicted, k=10):
    """
    Source: https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Source: https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def mr(actual, predicted):
    """
    Computes the recall of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        recall: int
    """
    def calc_recall(predicted, actual):
        reca = [value for value in predicted if value in actual]
        reca = np.round(float(len(reca)) / float(len(actual)), 4)
        return reca

    recall = np.mean(list(map(calc_recall, predicted, actual)))
    return recall

def evaluate_model(R_hat, test_mat, masked_test_mat, top_n):
    top_N_recs = np.array(np.argsort(-R_hat, axis=1)[:,:top_n]).tolist()
    y_true = sparse_to_list(test_mat)    
    
    MAP = mapk(y_true, top_N_recs)
    rec_at_k = mr(y_true, top_N_recs)
    
    del y_true, top_N_recs

    mpr_all = m_p_r(test_mat, R_hat, verbose=False)
    mpr_mask = m_p_r(masked_test_mat, R_hat, verbose=False)

    return MAP, rec_at_k, mpr_all, mpr_mask

def subgraph(user_id, graph_data, num_neg):
    """given subset of users create sub-graph dataset"""
    
    edge_id = torch.nonzero(torch.tensor([1 if i in user_id else 0 for i in graph_data.edge_user]), as_tuple=True)[0]
    
    # picks out subset of edges
    edge_user = graph_data.edge_user[edge_id]
    edge_item = graph_data.edge_item[edge_id]

    # collect pre computed node normalisations 
    edge_norm = graph_data.edge_norm[torch.cat((edge_item, edge_user), 0)]

    users = torch.unique(edge_user, sorted=True)
    items = torch.unique(edge_item, sorted=True)

    num_user = len(users)
    num_item = int(graph_data.num_items)
    num_edge = len(edge_id)
    # num_node = num_user+num_item

    # re map numbers
    user_dict = dict(zip(users.tolist(),[i for i in range(num_user)]))
    item_dict = dict(zip(items.tolist(),torch.add(items,-int(graph_data.num_users)).tolist()))

    # create edge index of shape 2 x (2*n_edges), counted twice since undirected 
    edge_index = torch.stack((torch.cat((edge_user, edge_item), 0),
                            torch.cat((edge_item, edge_user), 0)), 0)
    edge_index = edge_index.to(torch.long)

    x = graph_data.x

    # Prepare data
    sub_data = Data(x=x, edge_index=edge_index)
    sub_data.edge_norm = edge_norm        
    sub_data.num_users = torch.tensor([num_user])
    sub_data.num_items = torch.tensor([num_item])
    sub_data.users = users
    sub_data.items = items

    train_idx = [user_dict[int(edge_index[0,i])]*num_item+item_dict[int(edge_index[1,i])] for i in range(num_edge)]

    if num_neg>0:
        n_neg = int(num_neg*num_edge)
        
        full_neg_idx = list(set([i for i in range(num_user*num_item)])-set(train_idx))
        neg_idx = random.sample(full_neg_idx, n_neg)

        perm = torch.randperm((num_neg+1)*num_edge).long()
        sub_data.train_idx = torch.tensor(train_idx + neg_idx).long()[perm]
        sub_data.train_rt = torch.cat((torch.ones(num_edge), torch.zeros(n_neg)),0)[perm]

        del n_neg, full_neg_idx, neg_idx, perm
    else:
        sub_data.train_idx = torch.tensor(train_idx)
        sub_data.train_rt = torch.ones(num_edge)

    return sub_data
