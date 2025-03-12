import argparse
from os.path import isfile
from types import SimpleNamespace

import yaml
import os.path as osp


def parseDataArgs(configPath=None):
    parser = argparse.ArgumentParser()
    #加入参数保存位置
    parser.add_argument(
        '--config',
        type=str,
        default=configPath
    )
    parser.add_argument('--phi',type=float)
    parser.add_argument('--epsilon',type=float)
    parser.add_argument('--root',type=str)
    parser.add_argument('--name',type=str)
    parser.add_argument('--num_nodes',type=int)
    parser.add_argument('--num_features',type=int)
    parser.add_argument('--avg_degree',type=float)
    return parser.parse_args()

def parseTrainArgs(configPath=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=configPath
    )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--early_stopping', type=int)
    parser.add_argument('--hidden', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--train_rate', type=float)
    parser.add_argument('--val_rate', type=float)
    parser.add_argument('--K', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--dprate', type=float)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'])
    parser.add_argument('--Gamma')
    parser.add_argument('--ppnp',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', type=int)
    parser.add_argument('--output_heads', type=int)

    parser.add_argument('--dataset')
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--RPMAX', type=int)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN']
                        )
    return parser.parse_args()


def loadConfig(ymlPath=None):
    if ymlPath is None or isfile(ymlPath):
        print(f"warning: config file not found at {ymlPath}!")
        return {}
    current_dir = osp.dirname(osp.abspath(__file__))
    config_path = osp.join(current_dir, ymlPath)
    config_path = osp.normpath(config_path)
    #print(config_path)

    with open(config_path,'r',encoding='utf-8') as ymlFile:
        config = yaml.safe_load(ymlFile)
    return config or {}

def MergeDataConfig(args=None,config=None):
    if args.phi is not None:
        config['phi'] = args.phi
    if args.epsilon is not None:
        config['epsilon'] = args.epsilon
    if args.root is not None:
        config['root'] = args.root
    if args.name is not None:
        config['name'] = args.name
    if args.num_nodes is not None:
        config['num_nodes'] = args.num_nodes
    if args.num_features is not None:
        config['num_features'] = args.num_features
    if args.avg_degree is not None:
        config['avg_degree'] = args.avg_degree
    return config

def MergeTrainConfig(args=None,config=None):
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.lr is not None:
        config['lr'] = args.lr
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.early_stopping is not None:
        config['early_stopping'] = args.early_stopping
    if args.hidden is not None:
        config['hidden'] = args.hidden
    if args.dropout is not None:
        config['dropout'] = args.dropout
    if args.K is not None:
        config['K'] = args.K
    if args.alpha is not None:
        config['alpha'] = args.alpha
    if args.dprate is not None:
        config['dprate'] = args.dprate
    if args.C is not None:
        config['C'] = args.C
    if args.Init is not None:
        config['Init'] = args.Init
    if args.Gamma is not None:
        config['Gamma'] = args.Gamma
    if args.ppnp is not None:
        config['ppnp'] = args.ppnp
    if args.heads is not None:
        config['heads'] = args.heads
    if args.output_heads is not None:
        config['output_heads'] = args.output_heads
    if args.net is not None:
        config['net'] = args.net
    if args.dataset is not None:
        config['dataset'] = args.dataset
    if args.cuda is not None:
        config['cuda'] = args.cuda
    if args.RPMAX is not None:
        config['RPMAX'] = args.RPMAX
    if args.train_rate is not None:
        config['train_rate'] = args.train_rate
    if args.val_rate is not None:
        config['val_rate'] = args.val_rate
    return config

def loadAndMergeConfig(type=None,configPath=None):
    if type is None:
        print(f"the load config type is required")
        return {}
    elif type=='data':
        args = parseDataArgs(configPath)
        config = loadConfig(args.config)
        return SimpleNamespace(**MergeDataConfig(args,config))
    elif type=='train':
        args = parseTrainArgs(configPath)
        config=loadConfig(args.config)
        return SimpleNamespace(**MergeTrainConfig(args,config))
    else:
        print(f"type has to be 'data' or 'train'")
