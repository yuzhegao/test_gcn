import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse import block_diag

import torch
import torch.utils.data as data

from utils import normalize,sparse_mx_to_torch_sparse_tensor


def load_off(filename):
    vertice_list = []
    face_list = []
    f = open(filename, 'r')
    lines = f.readlines()

    if len(lines[0].rstrip().split()) == 1:
        num_v, num_f, _ = lines[1].rstrip().split()
        for i in range(2, 2 + int(num_v)):
            line = lines[i].rstrip().split()
            vertice_list.append(np.array([float(line[0]), float(line[1]), float(line[2])]))
        for i in range(2 + int(num_v), 2 + int(num_v) + int(num_f)):
            line = lines[i].rstrip().split()
            face_list.append(np.array([int(line[1]), int(line[2]), int(line[3])]))
    else:
        ## for the case: some wrong .off file  ,like "OFF1568 1820 0" in first row
        num_v, num_f, _ = lines[0].rstrip().split()
        num_v = num_v[3:]
        for i in range(1, 1 + int(num_v)):
            line = lines[i].rstrip().split()
            vertice_list.append(np.array([float(line[0]), float(line[1]), float(line[2])]))
        for i in range(1 + int(num_v), 1 + int(num_v) + int(num_f)):
            line = lines[i].rstrip().split()
            face_list.append(np.array([int(line[1]), int(line[2]), int(line[3])]))

    return int(num_v),int(num_f),np.array(vertice_list),np.array(face_list)

def load_data(graph_file,pts_file):
    """Load pts dataset (graph and pts feature)n for single shape"""

    ## pts feature
    nv,_,pts,_ = load_off(pts_file)
    features = sp.csr_matrix(pts, dtype=np.float32) ## sparse object [N,3]

    ## graph
    edges = []
    with open(graph_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split()
            for idx in line[1:]:
                edges.append([line[0], idx])
    edges = np.array(edges, dtype=np.int)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(nv, nv),
                        dtype=np.float32) ##(2708,2708)  -->the origin adjacent matrix (not symmetric)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) ## mutiply -> point-wise

    features = normalize(features)  ## in row, normalize
    adj = normalize(adj + sp.eye(adj.shape[0])) ## add identity, then normalize by row(devide by degree?)

    features = torch.FloatTensor(np.array(features.todense()))
    #adj = sparse_mx_to_torch_sparse_tensor(adj)

    ## feature:dense  adj:sparse
    return adj, features

## torch class
class graph_dataset(data.Dataset):
    def __init__(self,data_root,dir_list_file='dir_list.txt',is_training = True):
        super(graph_dataset, self).__init__()

        self.data_root = data_root
        self.feat_path = os.path.join(data_root,'ModelNet40')
        self.adj_path = os.path.join(data_root,'GCN_ModelNet40')
        with open(os.path.join(data_root,dir_list_file)) as f:
            dir_list = f.readlines()

        label = 0
        suffix = 'train' if is_training else 'test'

        self.label_list = []
        self.filename_list = []
        for dir in dir_list:
            dir = dir.rstrip()
            if not dir.endswith(suffix):
                continue
            else:
                cls_dir = os.path.join(data_root, 'ModelNet40', dir)
                files = os.listdir(cls_dir)
                for fl in files:
                    self.filename_list.append(os.path.join(dir, fl)[:-4])
                labels = np.full((len(files),), fill_value=label)
                self.label_list.append(labels)
                label += 1
        self.label_list = np.concatenate(self.label_list, axis=0)

    def __getitem__(self, idx):
        filename = self.filename_list[idx]
        label = self.label_list[idx]
        pts_file, adj_file = os.path.join(self.feat_path,filename) + '.off',\
                             os.path.join(self.adj_path,filename) + '.txt'
        adj, features = load_data(adj_file,pts_file)

        return adj, features, label

    def __len__(self):
        return len(self.label_list)


def gcn_collate(batch):
    num_pts = []
    adj_batch = []
    feat_batch = []
    label_batch = []

    for sample in batch:
        adj_batch.append(sample[0])
        num_pts.append(len(sample[1]))
        feat_batch.append(sample[1])
        label_batch.append(sample[2])

    adj_batch = block_diag(adj_batch)
    adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch)
    feat_batch = torch.cat(feat_batch, dim=0)

    label_batch =torch.from_numpy(np.squeeze(label_batch))

    return adj_batch,feat_batch.float(),label_batch.long(),num_pts


if __name__ == '__main__':

    my_dataset = graph_dataset(data_root='/home/yuzhe/Downloads/3d_data/test_gcn',dir_list_file='dir_list.txt',is_training=False)
    print len(my_dataset)
    print my_dataset.filename_list[0]

    loader = torch.utils.data.DataLoader(my_dataset, batch_size=2, shuffle=True, collate_fn=gcn_collate)

    for idx, (adj, features, label, num_pts) in enumerate(loader):
        #print (adj.size())
        #print (features.size())
        #print (label.size())
        print num_pts
        adj1 = adj
        features1 = features
        break


    from models.models import GCN

    model = GCN(nfeat=features1.shape[1],
                nclass=40,
                dropout_rate=0.5)

    if torch.cuda.is_available():
        model.cuda()
        features = features1.cuda()
        adj = adj1.cuda()

    model.train()
    output = model(features1, adj1, num_pts)
    print
    print output.size()



    """

    ## test effcient with a large adjcent matrix
    ## result: very fast
    adj1, features1,_ = my_dataset[0] #load_data('../data/airplane_0627.txt', '../data/airplane_0627.off')
    adj2, features2,_ = my_dataset[10] #load_data('../data/desk_0201.txt', '../data/desk_0201.off')
    print adj1.shape, adj2.shape
    print features1.shape,features2.shape
    adj=block_diag([adj1,adj2,adj1,adj1]) ##(40200, 40200)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = torch.cat([features1,features2,features1,features1], dim=0)
    print adj.shape
    """