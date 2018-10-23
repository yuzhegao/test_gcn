import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout_rate):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 64)
        self.gc2 = GraphConvolution(64, 64)
        self.gc3 = GraphConvolution(64, 1024)
        self.dropout_rate = dropout_rate
        self.mlp = nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 40)
        )

    def forward(self, x, adj, num_pts):
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout_rate, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))  ## [N1+N2+N3, 1024]

        global_feat = []
        shapes_feat = torch.split(x,num_pts,dim=0)
        for feat in shapes_feat:
            #print torch.max(feat,dim=0)[0]
            global_feat.append(torch.max(feat,dim=0)[0])

        global_feat = torch.stack(global_feat, dim=0)
        #print global_feat.size()
        scores = self.mlp(global_feat)
        pred = F.log_softmax(scores, dim=-1)

        return pred

