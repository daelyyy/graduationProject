import json
import os
from datetime import datetime
from itertools import product

from model.APPNP import APPNP_Net
from model.Bern import BernNet
from model.ChebNet import Cheb_Net
from model.ChebNet2 import ChebNetII
from model.GAT import GAT_Net
from model.GCN import GCN_Net
from model.GNNHF import GNNsHF
from model.GNNLF import GNNsLF
from model.GPRGNN import GPR
from model.JKNet import GCN_JKNet
from model.MLP import MLP_
from model.SGC import SGC_Net
from model.SSGC import SSGC_Net
from utils.LoadDataset_utils import load_dataset, sgc_precompute

import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import utils.Config_utils as Config
from utils.Random_utils import random_planetoid_splits

def RunExp(args, dataset, data, Net, percls_trn, val_lb):
    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    return test_acc, best_val_acc, Gamma_0


if __name__ == '__main__':
    sgc_degree = 3

    args = Config.loadAndMergeConfig('train','../config/trainConfig.yml')

    # 超参数列表
    alpha_values = [0.1, 0.2, 0.5, 0.9]
    lr_values = [0.002, 0.01, 0.05]
    weight_decay_values = [0.0005]
    # dataset_values = ['cSBM_-0.5','cSBM_-1','cSBM_-0.75','cSBM_-0.25','cSBM_0.0','cSBM_0.5','cSBM_0.25','cSBM_0.75','cSBM_1']
    dataset_values = ['Chameleon']
    # 在这里生成所有的参数组合
    param_combinations = list(product(alpha_values, lr_values,weight_decay_values,dataset_values))

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = Cheb_Net
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPR
    elif gnn_name == 'SSGC':
        Net = SSGC_Net
    elif gnn_name == 'SGC':
        Net = SGC_Net
    elif gnn_name == 'MLP':
        Net = MLP_
    elif gnn_name == 'ChebNetII':
        Net = ChebNetII
    elif gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name == 'GNN-LF':
        Net = GNNsLF
    elif gnn_name == 'GNN-HF':
        Net = GNNsHF

    best_res_log_filename=''
    best_results = {}  # 用于保存最佳参数
    for alpha,lr,weight_decay_values,dataset_values in tqdm(param_combinations):
        print(f"Running experiment with alpha={alpha} lr={lr} weight_decay_values={weight_decay_values} dataset_values={dataset_values}")
        args.alpha = alpha
        args.lr = lr
        args.dataset = dataset_values
        args.weight_decay = weight_decay_values

        dname = args.dataset
        dataset, data = load_dataset(dname)

        RPMAX = args.RPMAX
        Init = args.Init

        # gamma和c是cSBM数据集中药用到的参数，其他数据集可忽略
        Gamma_0 = None
        train_rate = args.train_rate
        val_rate = args.val_rate
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
        print('True Label rate: ', TrueLBrate)

        args.C = len(data.y.unique())
        args.Gamma = Gamma_0

        Results0 = []
        if Net == SGC_Net:
            adj = data.edge_index
            features = data.x
            feature, time = sgc_precompute(features, adj, sgc_degree)
            data.x = feature

        log_dir = "./logs"
        result_dir="./results"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        timestamp = datetime.now().strftime('%m%d_%H%M%S')  # 例如：20231126_144500
        log_filename = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        best_res_log_filename = os.path.join(result_dir, f'best_{dname}_{timestamp}_{gnn_name}.txt')

        for RP in tqdm(range(RPMAX)):
            test_acc, best_val_acc, Gamma_0 = RunExp(
                args, dataset, data, Net, percls_trn, val_lb)
            Results0.append([test_acc, best_val_acc])
            #Results0.append([test_acc, best_val_acc, Gamma_0])

        test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
        test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
        with open(log_filename, 'a') as log_file:  # 以追加模式写入
            log_file.write(
                f"{dname},{gnn_name},{alpha},{args.lr},{args.weight_decay},{train_rate},{val_rate},{val_acc_mean:.4f},{test_acc_mean:.4f}\n")
        print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
        print(
            f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')

        if dname not in best_results:
            best_results[dname] = {
                'alpha': alpha,
                'lr': args.lr,
                'test_acc': test_acc_mean
            }
        elif dname in best_results and test_acc_mean > best_results[dname]['test_acc']:
            best_results[dname]['test_acc'] = test_acc_mean
            best_results[dname]['alpha'] = alpha
            best_results[dname]['lr'] = args.lr

    with open(best_res_log_filename, 'w') as f:
        json.dump(best_results, f, indent=4)
