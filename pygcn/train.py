from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN

def train(epoch,model, optimizer, features, adj, labels, idx_train, idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, features, adj, labels, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

def main():
    # Training settings
    param = {
        'no_cuda': False, #set True if you don't have a GPU
        'seed': 42, #random seed for reproducibility
        'epochs': 200, #Number of epochs to train for
        'lr': 0.01, #Learning rate
        'weight_decay': 5e-4, #Weight decay (L2 loss on parameters)
        'hidden': 16, #Number of hidden units
        'dropout': 0.5 #Dropout rate
    }

    cuda = not param['no_cuda'] and torch.cuda.is_available()

    np.random.seed(param['seed'])
    torch.manual_seed(param['seed'])
    if cuda:
        torch.cuda.manual_seed(param['seed'])

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=param['hidden'],
                nclass=labels.max().item() + 1,
                dropout=param['dropout'])
    optimizer = optim.Adam(model.parameters(),
                        lr=param['lr'], weight_decay=param['weight_decay'])

    if cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    # Train model
    t_total = time.time()
    for epoch in range(param['epochs']):
        train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val)
    print("Training Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test(model, features, adj, labels, idx_test)

main()
    