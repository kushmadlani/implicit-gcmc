import torch
import yaml
import scipy.sparse as sp

from dataset import MCDataset, MCBatchDataset
from model import GAE
from model_batch import GAE_Batch
from trainer import Trainer
from utils import calc_rmse, ster_uniform, random_init, init_xavier, init_uniform, Config, evaluate_model

import wandb

import os
os.environ['WANDB_API_KEY'] = 'f2935f67cdf03fb8a19d09e8bc7124024891dc3f'

def main(cfg):
    wandb.init(project='gcmc_bets_test')

    cfg = Config(cfg)

    # device and dataset setting
    use_gpu = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(use_gpu)
    # device = (torch.device(f'cuda:{cfg.gpu_id}')
    #     if torch.cuda.is_available() and cfg.gpu_id >= 0
    #     else torch.device('cpu'))
    cfg.device = device
    cfg.use_gpu = use_gpu
    print(device)

    if cfg.batch_ratio>0:
        dataset = MCBatchDataset(cfg.root, cfg.dataset_name, cfg.n_items)
    else:
        dataset = MCDataset(cfg.root, cfg.dataset_name, cfg.num_neg, cfg.n_items)
    # data = dataset[0].to(device)
    data = dataset[0]
    print(data)

    # add some params to config
    cfg.num_nodes = dataset.num_nodes
    cfg.num_users = int(data.num_users)


    # initialise experiment tracker
    wandb.config.update(cfg)

    # set and init model
    if cfg.batch_ratio>0:
        model = GAE_Batch(cfg, random_init).to(device)
    else:
        model = GAE(cfg, random_init).to(device)

    model.apply(init_xavier)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # load test data
    test_data = {
        'val': sp.load_npz(cfg.test_root+'val_unmasked.npz'),
        'val_masked': sp.load_npz(cfg.test_root+'val_masked.npz')
    }

    # train
    trainer = Trainer(
        model, data, optimizer, cfg, calc_eval=True, calc_metrics=evaluate_model, top_n=5, test_data=test_data,
        batch_ratio=cfg.batch_ratio, batch_neg=cfg.num_neg,
    )
    trainer.training(cfg.epochs)

if __name__ == '__main__':
    with open('config.yml') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
    # main(cfg, comet=True)