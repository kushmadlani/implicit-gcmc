import numpy as np
import pandas as pd 
import pickle
import datetime
import time
import scipy.sparse as sp
import matplotlib.pyplot as plt

import tarfile
import requests
import gzip
import copy
import os
from os import path


def new_test(R_train, R_test):
    ind = (R_train==0).nonzero()
    mask_array = sp.csr_matrix(R_test.shape)
    mask_array[ind] = True
    R_test_new = R_test.multiply(mask_array)
    return R_test_new

def download_preprocess(url, file_path, save_path, cols, n=3, m=5, test_ratio=0.2): 
    
    # download file if not exists
    if not path.exists(file_path):
        filename = url.split("/")[-1]
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
        tar = tarfile.open(filename, "r:gz")
        tar.extract(file_path)
        tar.close()
    
    df = pd.read_csv(file_path, sep="\t", names=cols, error_bad_lines=False)

    # remove null songs and items
    df = df[df.songId.notnull()]
    df = df[df.artistId.notnull()]

    # ensure each artist has at least n unique listeners
    artists = df.groupby('artistId')['userId'].nunique()
    artists = artists[artists>n].index.tolist()
    df = df.query('artistId in @artists')

    # ensure each user has at least m unique artists
    users = df.userId.value_counts()
    users = users[users>m].index.tolist()
    df = df.query('userId in @users')

    # create date column for train/test split
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df["date"] = [d.date() for d in df["timestamp"]]

    # re-index users and items
    users = list(df.userId.unique())
    user_dict = dict(zip(users,[i for i in range(len(users))]))
    df['userId'] = df['userId'].apply(lambda x: user_dict[x])
    items = list(df.artistId.unique())
    item_dict = dict(zip(items,[i for i in range(len(items))]))
    df['artistId'] = df['artistId'].apply(lambda x: item_dict[x])

    # create train + test split
    dates = sorted(list(df['date'].unique()))
    test_days = int(len(dates)*test_ratio)
    train_df = df[df['date'] <= dates[-test_days]]
    test_df = df[df['date'] > dates[-test_days]]

    # find users in both train and test
    train_users = train_df['userId'].unique()
    test_users = test_df['userId'].unique()
    valid_users = list(set(train_users)&set(test_users))

    # filter for users in both train and test
    train_df = train_df[train_df['userId'].isin(valid_users)]
    test_df = test_df[test_df['userId'].isin(valid_users)]

    # get list of items present in both 
    train_leagues = train_df['artistId'].unique()
    test_leagues = test_df['artistId'].unique()
    valid_leagues = list(set(train_leagues)|set(test_leagues))

    # get matrices
    train_df = train_df.groupby(['userId', 'artistId']).size().unstack(fill_value=0)
    test_df = test_df.groupby(['userId', 'artistId']).size().unstack(fill_value=0)

    # add missing columns to each
    for league in valid_leagues:
        if league not in train_df:
            train_df[league]=0
        if league not in test_df:
            test_df[league]=0

    # line up columns
    cols = list(train_df.columns.values)
    train_df = train_df.reindex(columns=cols)
    test_df = test_df.reindex(columns=cols)

    # turn df into sparse matrices
    train_mat = sp.csr_matrix(train_df.values)
    test_mat = sp.csr_matrix(test_df.values)
    # get masked test set
    masked_test_mat = new_test(train_mat, test_mat)

    # save test sparse matrices to file
    if not path.exists(save_path+'raw/'):
        os.makedirs(save_path+'raw/')
        os.makedirs(save_path+'test/')
    sp.save_npz(save_path+'test/masked.npz', masked_test_mat)
    sp.save_npz(save_path+'test/unasked.npz', test_mat)

    # get user-item-count triples for iGC-MC
    train_df['userId'] = train_df.index
    train_df = train_df.melt('userId', var_name='itemId', value_name='rating')
    train_df = train_df[train_df.rating != 0]

    # save training triplets dataframe to file
    train_df.to_pickle(save_path+'raw/train.pkl')