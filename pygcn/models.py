import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        #First layer
        x = self.gc1(x, adj)
        x = F.relu(x)
        #random dropout per layer to prevent overfitting
        x = F.dropout(x, self.dropout, training=self.training) 
        #Second layer
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
