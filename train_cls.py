from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from utils.utils import accuracy
from models.models import GCN
from utils.data_utils import graph_dataset,gcn_collate

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='DIR',default='/home/yuzhe/Downloads/3d_data/test_gcn')
parser.add_argument('--log', metavar='LOG',default='log_classification',help='dir of log file and resume')
parser.add_argument('--log-step', default=500, type=int, metavar='N',help='number of iter to write log')
parser.add_argument('--gpu', default=0, type=int, metavar='N',help='the index  of GPU where program run')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('-bs',  '--batch-size', default=2 , type=int,metavar='N', help='mini-batch size (default: 2)')
parser.add_argument('--epochs', type=int, default=200,help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_step', default=200000, type=int,
                    metavar='LR', help='decay_step of learning rate')
parser.add_argument('--decay_rate', default=0.7, type=float,
                    metavar='LR', help='decay_rate of learning rate')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--resume', default=None,type=str, metavar='PATH',help='path to latest checkpoint ')

args = parser.parse_args()
is_cuda = torch.cuda.is_available()

LOG_DIR = os.path.join(args.log,time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
print ('prepare training in {}'.format(time.strftime('%Y-%m-%d-%H:%M',time.localtime(time.time()))))

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if args.resume is None:
    resume = os.path.join(LOG_DIR, "checkpoint.pth")
else:
    resume = args.resume

logname = os.path.join(LOG_DIR,'log.txt')
optfile = os.path.join(LOG_DIR,'opt.txt')
with open(optfile, 'wt') as opt_f:
    opt_f.write('------------ Options -------------\n')
    for k, v in sorted(vars(args).items()):
        opt_f.write('%s: %s\n' % (str(k), str(v)))
    opt_f.write('-------------- End ----------------\n')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if is_cuda:
    torch.cuda.manual_seed(args.seed)

my_dataset = graph_dataset(data_root=args.data, is_training=True)
data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=8, shuffle=True, collate_fn=gcn_collate)

# Model and optimizer
model = GCN(nfeat=3,
            nclass=40,
            dropout_rate=args.dropout)
if is_cuda:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
critenrion = nn.NLLLoss()

def save_checkpoint(epoch,model,num_iter):
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'iter':num_iter,
    },resume)

def log(filename,epoch,batch,loss):
    f1=open(filename,'a')
    if epoch == 0 and batch == 0:
        f1.write("\nstart training in {}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    f1.write('\nin epoch{} batch{} loss={} '.format(epoch,batch,loss))
    f1.close()

"""
def evaluate(model_test):
    model_test.eval()
    total_correct=0

    data_eval =pts_cls_dataset(datalist_path=args.data_eval,data_argument=False,num_points=args.num_pts,use_extra_feature=args.normal)
    eval_loader = torch.utils.data.DataLoader(data_eval,
                    batch_size=4, shuffle=True, collate_fn=pts_collate)
    print ("dataset size:",len(eval_loader.dataset))

    for batch_idx, (pts, label) in enumerate(eval_loader):
        if is_GPU:
            pts = Variable(pts.cuda())
            label = Variable(label.cuda())
        else:
            pts = Variable(pts)
            label = Variable(label)
        pred,trans = net(pts)

        _, pred_index = torch.max(pred, dim=1)
        num_correct = (pred_index.eq(label)).data.cpu().sum().item()
        total_correct +=num_correct

    print ('the average correct rate:{}'.format(total_correct*1.0/(len(eval_loader.dataset))))

    model_test.train()
    with open(logname,'a') as f:
        f.write('\nthe evaluate average accuracy:{}'.format(total_correct*1.0/(len(eval_loader.dataset))))
"""

#########
## train
#########


def train():
    model.train()
    num_iter=0
    start_epoch=0

    if os.path.exists(resume):
        if is_cuda:
            checkoint = torch.load(resume)
        else:
            checkoint = torch.load(resume, map_location=lambda storage, loc: storage)
        start_epoch = checkoint['epoch']
        model.load = model.load_state_dict(checkoint['model'])
        num_iter= checkoint['iter']
        print ('load the resume checkpoint,train from epoch{}'.format(start_epoch))
    else:
        print("Warining! No resume checkpoint to load")

    print('start training')

    for epoch in xrange(start_epoch,args.epochs):
        init_epochtime = time.time()

        ##--------------------------------------------------------------
        for idx, (adj, features, label, num_pts) in enumerate(data_loader):
            t1=time.time()
            print (num_pts)
            if is_cuda:
                adj = Variable(adj.cuda())
                features = Variable(features.cuda())
                label = Variable(label.cuda())
            else:
                adj = Variable(adj)
                features = Variable(features)
                label = Variable(label)

            pred = model(features,adj,num_pts)
            loss = critenrion(pred, label)

            _, pred_index = torch.max(pred, dim=1)
            num_correct = (pred_index.eq(label)).data.cpu().sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t2=time.time()
            num_iter+=1

            print('In Epoch{} Iter{},loss={} accuracy={}  time cost:{}'.format(epoch,num_iter, loss.data,num_correct.item() / args.batch_size,t2-t1))
            if num_iter%(args.log_step*10)==0 and num_iter!=0:
                save_checkpoint(epoch, model, num_iter)
                #evaluate(model)
            if num_iter%(args.log_step)==0 and num_iter!=0:
                log(logname, epoch, num_iter, loss.data)

            if (num_iter*args.batch_size)%args.decay_step==0 and num_iter!=0:
                f1 = open(logname, 'a')
                f1.write("learning rate decay in iter{}\n".format(num_iter))
                f1.close()
                print ("learning rate decay in iter{}\n".format(num_iter))
                for param in optimizer.param_groups:
                    param['lr'] *= args.decay_rate
                    param['lr'] = max(param['lr'],0.00001)

        end_epochtime = time.time()
        print('--------------------------------------------------------')
        print('in epoch:{} use time:{}'.format(epoch, end_epochtime - init_epochtime))
        print('-------------------------------------------------------- \n')

    save_checkpoint(args.epochs-1, model, num_iter)
    #evaluate(model)

train()

"""
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == '__main__':
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
#test()
"""
