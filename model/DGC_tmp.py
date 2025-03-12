import time
import yaml
import torch

import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F

from tqdm.notebook import tqdm
from torch.optim import Adam, Optimizer
from collections import defaultdict
from torch_geometric.data import Data, InMemoryDataset

from data import get_dataset, HeatDataset, PPRDataset, set_train_val_test_split
from models import GCN
from seeds import val_seeds, test_seeds


def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]

    # Self-loops
    A_loop = sp.eye(N) + A

    # Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec

    return T_S

device = 'cpu'
with open('config.yaml', 'r') as c:
    config = yaml.safe_load(c)

datasets = {}

for preprocessing in ['none', 'heat', 'ppr']:
    if preprocessing == 'none':
        dataset = get_dataset(
            name=config['dataset_name'],
            use_lcc=config['use_lcc']
        )
        dataset.data = dataset.data.to(device)
        datasets[preprocessing] = dataset
    elif preprocessing == 'heat':
        dataset = HeatDataset(
            name=config['dataset_name'],
            use_lcc=config['use_lcc'],
            t=config[preprocessing]['t'],
            k=config[preprocessing]['k'],
            eps=config[preprocessing]['eps']
        )
        dataset.data = dataset.data.to(device)
        datasets[preprocessing] = dataset
    elif preprocessing == 'ppr':
        dataset = PPRDataset(
            name=config['dataset_name'],
            use_lcc=config['use_lcc'],
            alpha=config[preprocessing]['alpha'],
            k=config[preprocessing]['k'],
            eps=config[preprocessing]['eps']
        )
        dataset.data = dataset.data.to(device)
        datasets[preprocessing] = dataset

models = {}

for preprocessing, dataset in datasets.items():
    models[preprocessing] = GCN(
        dataset,
        hidden=config[preprocessing]['hidden_layers'] * [config[preprocessing]['hidden_units']],
        dropout=config[preprocessing]['dropout']
    ).to(device)


def train(model: torch.nn.Module, optimizer: Optimizer, data: Data):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def evaluate(model: torch.nn.Module, data: Data, test: bool):
    model.eval()
    with torch.no_grad():
        logits = model(data)
    eval_dict = {}
    keys = ['val', 'test'] if test else ['val']
    for key in keys:
        mask = data[f'{key}_mask']
        # loss = F.nll_loss(logits[mask], data.y[mask]).item()
        # eval_dict[f'{key}_loss'] = loss
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        eval_dict[f'{key}_acc'] = acc
    return eval_dict


def run(dataset: InMemoryDataset,
        model: torch.nn.Module,
        seeds: np.ndarray,
        test: bool = False,
        max_epochs: int = 10000,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        num_development: int = 1500,
        device: str = 'cuda'):
    start_time = time.perf_counter()

    best_dict = defaultdict(list)

    cnt = 0
    for seed in tqdm(seeds):
        dataset.data = set_train_val_test_split(
            seed,
            dataset.data,
            num_development=num_development,
        ).to(device)
        model.to(device).reset_parameters()
        optimizer = Adam(
            [
                {'params': model.non_reg_params, 'weight_decay': 0},
                {'params': model.reg_params, 'weight_decay': weight_decay}
            ],
            lr=lr
        )

        patience_counter = 0
        tmp_dict = {'val_acc': 0}

        for epoch in range(1, max_epochs + 1):
            if patience_counter == patience:
                break

            train(model, optimizer, dataset.data)
            eval_dict = evaluate(model, dataset.data, test)

            if eval_dict['val_acc'] < tmp_dict['val_acc']:
                patience_counter += 1
            else:
                patience_counter = 0
                tmp_dict['epoch'] = epoch
                for k, v in eval_dict.items():
                    tmp_dict[k] = v

        for k, v in tmp_dict.items():
            best_dict[k].append(v)

    best_dict['duration'] = time.perf_counter() - start_time
    return dict(best_dict)

results = {}

for preprocessing in ['none', 'heat', 'ppr']:
    results[preprocessing] = run(
        datasets[preprocessing],
        models[preprocessing],
        seeds=test_seeds if config['test'] else val_seeds,
        lr=config[preprocessing]['lr'],
        weight_decay=config[preprocessing]['weight_decay'],
        test=config['test'],
        num_development=config['num_development'],
        device=device
    )

for preprocessing in ['none', 'heat', 'ppr']:
    mean_acc = results[preprocessing]['test_acc']
    uncertainty = results[preprocessing]['test_acc_ci']
    print(f"{preprocessing}: Mean accuracy: {100 * mean_acc:.2f} +- {100 * uncertainty:.2f}%")