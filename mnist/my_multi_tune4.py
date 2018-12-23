#!/usr/bin/env python
# coding=utf-8

from my_multi_main3 import main
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--norm-flag', type=bool, default=False,
                    help='Triggering the Layer Normalization flag for attention scores')
parser.add_argument('--gamma', type=float, default=None,
                    help='Controlling the sparisty of gfusedmax/sparsemax, the smaller, the more sparse')
parser.add_argument('--lam', type=float, default=1.0,
                    help='Lambda: Controlling the smoothness of gfusedmax, the larger, the smoother')
parser.add_argument('--max-type', type=str, default='softmax',choices=['softmax','sparsemax','gfusedmax'],
                    help='mapping function in attention')
parser.add_argument('--optim-type', type=str, default='SGD',choices=['SGD','Adam'],
                    help='mapping function in attention')
parser.add_argument('--head-cnt', type=int, default=2, metavar='S', choices=[1,2,4,5,10],
                    help='Number of heads for attention (default: 1)')

args = parser.parse_args()

hyperparameter_choices = {
    'lr':list(10**np.arange(-4,-1,0.5)),
    'norm_flag': [True,False],
    'gamma':list(10**np.arange(-1,3,0.5))+[None,],
    'lam':list(10**np.arange(-2,2,0.5)),
    'max_type':['softmax','sparsemax','gfusedmax'],
    # 'max_type':['sparsemax'],
    'optim_type':['SGD','Adam'],
    'head_cnt':[1,2,4,5,10,20]
}

param_num = 25
record = np.zeros([param_num,len(hyperparameter_choices)+1])
record_name = 'record4_multi_%s.csv'%time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime())
np.savetxt(record_name, record, delimiter=',')
for n in range(param_num):
    for param_index,(k,v) in enumerate(hyperparameter_choices.items()):
        print(param_index,k)
        value_index = np.random.choice(len(v))
        if k == 'max_type':
            value_index = 2
        if isinstance(v[value_index],str) or isinstance(v[value_index],bool) or v[value_index] is None:
            record[n,param_index] = value_index
        else:
            record[n,param_index] = v[value_index]
        setattr(args,k,v[value_index])
    record[n,-1] = main(args)
    np.savetxt(record_name, record, delimiter=',')



