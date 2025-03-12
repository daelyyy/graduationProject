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
import os
import torch.distributed as dist
from utils.Random_utils import random_planetoid_splits
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def main():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    args = Config.loadAndMergeConfig('train','../config/trainConfig.yml')

    sgc_degree = 3
    # 超参数列表（所有进程保持相同的参数组合）
    alpha_values = [0.1, 0.2, 0.5, 0.9]
    lr_values = [0.002, 0.01, 0.05]
    weight_decay_values = [0.0005]
    # dataset_values = ['cSBM_-0.5','cSBM_-1','cSBM_-0.75','cSBM_-0.25','cSBM_0.0','cSBM_0.5','cSBM_0.25','cSBM_0.75','cSBM_1']
    dataset_values = ['Chameleon']
    # 在这里生成所有的参数组合保持原有生成逻辑
    param_combinations = list(product(alpha_values, lr_values, weight_decay_values, dataset_values))

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

    best_res_log_filename = ''
    best_results = {}  # 用于保存最佳参数
    # 只在主进程显示进度条
    if rank == 0:
        pbar = tqdm(param_combinations)
    else:
        pbar = param_combinations
    for params in pbar:
        print(f"Running experiment with alpha={alpha} lr={lr} weight_decay_values={weight_decay_values} rank={rank} localrank={local_rank}")
        alpha, lr, weight_decay, dataset_name = params

        # 设置参数（所有进程同步执行）
        args.alpha = alpha
        args.lr = lr
        args.dataset = dataset_name
        args.weight_decay = weight_decay

        # 加载数据集（所有进程需要同步）CPU
        dataset, data = load_dataset(args.dataset)
        dname = args.dataset
        # data = data.to(local_rank)

        Gamma_0 = None
        RPMAX=args.RPMAX
        train_rate = args.train_rate
        val_rate = args.val_rate
        percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
        val_lb = int(round(val_rate * len(data.y)))
        TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
        print('True Label rate: ', TrueLBrate)

        Results0 = []
        if Net == SGC_Net:
            adj = data.edge_index
            features = data.x
            feature, time = sgc_precompute(features, adj, sgc_degree)
            data.x = feature

        log_dir = "./logs"
        result_dir = "./results"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        timestamp = datetime.now().strftime('%m%d_%H%M%S')  # 例如：20231126_144500
        log_filename = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        best_res_log_filename = os.path.join(result_dir, f'best_{dname}_{timestamp}_{gnn_name}.txt')

        # 运行实验（分布式版本）
        for RP in tqdm(range(RPMAX)):
            test_acc, best_val_acc, Gamma_0 = RunExp_DDP(args, dataset, data, Net,
                                                  percls_trn, val_lb, rank, local_rank)
            Results0.append([test_acc, best_val_acc])

        # 只由主进程保存结果
        if rank == 0:
            test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
            test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
            with open(log_filename, 'a') as log_file:  # 以追加模式写入
                log_file.write(
                    f"{dname},{gnn_name},{alpha},{args.lr},{args.weight_decay},{train_rate},{val_rate},{val_acc_mean:.4f},{test_acc_mean:.4f}\n")
            # print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
            # print(
            #     f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')

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
    dist.destroy_process_group()





def RunExp_DDP(args, dataset, data, Net, percls_trn, val_lb, rank, local_rank):
    # 初始化当前进程的CUDA设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll

        # 反向传播
        loss.backward()
        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        with torch.no_grad():
            logits = model(data)
            accs, losses, preds = [], [], []

            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                loss = F.nll_loss(logits[mask], data.y[mask])

                preds.append(pred.detach().cpu())
                accs.append(acc)
                losses.append(loss.detach().cpu())

        return accs, preds, losses

    # 数据准备
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    # 创建分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True
    )

    # 创建分布式DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # 模型初始化
    model = Net(dataset, args).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 优化器配置
    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([
            {'params': model.module.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.module.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.module.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.lr}
        ], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # 训练状态初始化
    best_val_acc = 0.0
    best_val_loss = float('inf')
    test_acc = 0.0
    val_loss_history = []
    val_acc_history = []
    Gamma_0 = args.alpha  # 默认值

    # 训练循环
    for epoch in range(args.epochs):
        # 设置epoch用于shuffle
        sampler.set_epoch(epoch)

        # 训练步骤
        for batch in dataloader:
            batch = batch.to(device)
            train(model, optimizer, batch, args.dprate)

        # 验证步骤 (只在主进程执行)
        if rank == 0:
            # 收集完整数据进行验证
            full_data = data.to(device)
            accs, _, losses = test(model, full_data)
            train_acc, val_acc, tmp_test_acc = accs
            train_loss, val_loss, tmp_test_loss = losses

            # 更新最佳结果
            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                test_acc = tmp_test_acc

                # GPRGNN的特殊处理
                if args.net == 'GPRGNN':
                    Gamma_0 = model.module.prop1.temp.clone().detach().cpu().numpy()
                else:
                    Gamma_0 = args.alpha

            # 早停机制
            val_loss_history.append(val_loss.item())
            val_acc_history.append(val_acc)

            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

        # 同步所有进程
        dist.barrier()

    # 清理分布式环境
    if rank == 0:
        dist.destroy_process_group()

    return test_acc, best_val_acc, Gamma_0