import torch
import yaml
import scipy.sparse as sp
import numpy as np
import pickle 
import argparse 
import wandb
import os
# os.environ['WANDB_API_KEY'] = # insert weights&biases key to track experiments

from dataset import MCDataset, MCBatchDataset
from model import GAE
from model_batch import GAE_Batch
from trainer import Trainer, TrainerBatch
from utils import get_args, init_xavier, random_init, evaluate_model

def main(cfg):
    wandb.init(project=cfg.project)

    # device and dataset setting
    use_gpu = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(use_gpu)
    cfg.device = device
    cfg.use_gpu = use_gpu
    batch = True if cfg.batch_ratio>0 else False
    print(device)
    
    # load side information
    if cfg.item_side_info:
        side_info = {}
        side_info['data'] = torch.tensor(np.load(cfg.side_root+'info2.npy'))
        cfg.n_side_features = side_info['data'].size(1)
    else:
        side_info = None

    if batch:
        dataset = MCBatchDataset(cfg.root, cfg.dataset, cfg.num_items)
    else:
        dataset = MCDataset(cfg.root, cfg.dataset, cfg.num_neg, cfg.num_items)

    data = dataset[0]
    print(data)

    # add some params to config
    cfg.num_nodes = dataset.num_nodes
    cfg.num_users = int(data.num_users)

    # initialise experiment tracker
    wandb.config.update(cfg)

    # set and init model
    if batch:
        model = GAE_Batch(cfg, random_init, side_info).to(device)
    else:
        model = GAE(cfg, random_init, side_info).to(device)
    model.apply(init_xavier)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # load test data
    test_data = {
        'test': sp.load_npz(cfg.test_root+'unmasked.npz'),
        'test_masked': sp.load_npz(cfg.test_root+'masked.npz')
    }

    # create trainer 
    if batch:
        trainer = TrainerBatch(
            model, data, optimizer, cfg, calc_eval=cfg.calc_eval, calc_metrics=evaluate_model, top_n=5, val_data=test_data,
            batch_ratio=cfg.batch_ratio, batch_neg=cfg.num_neg,
        )   
    else:
        trainer = Trainer(
            model, data, optimizer, cfg, calc_eval=cfg.calc_eval, calc_metrics=evaluate_model, top_n=5, val_data=test_data
        )
    # train model
    trainer.training(cfg.epochs)


if __name__ == '__main__':
    cfg = get_args()
    wandb.init(project=cfg.project)    
    main(cfg)
