from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import random
import torch
from torch import nn
import dgl
import torch.nn.functional as F
import utils
from model import DGASN
from flip_gradient import GradReverse
from utils import  GET_AUC_PR, GET_AUC_ROC, \
    edge_prepare, normalize_features
from sklearn.metrics import average_precision_score


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')

parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr-ini', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--l2-w', type=float, default=0.001,
                    help='weight of L2-norm regularization')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--num-heads", type=int, default=8,
                    help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=8,
                    help="number of hidden layers")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-hidden", type=int, default=64,
                    help="number of hidden units")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=.6,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.1,
                    help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")

parser.add_argument('--Hid-dim-edge-clf', type=int, default=128,
                    help='Dimensions of hidden layers of edge classification.')
parser.add_argument('--Hid-dim-node-clf', type=int, default=32,
                    help='Dimensions of hidden layers of node classification.')
parser.add_argument('--Hid-dim-edge-domain-1', type=int, default=128,
                    help='Dimensions of the first hidden layers of domain classification.')
parser.add_argument('--Hid-dim-edge-domain-2', type=int, default=32,
                    help='Dimensions of the second hidden layers of domain classification.')

parser.add_argument('--edge-type', type=str, default='concat',
                    help='concat,avg,had,L1 or L2')
parser.add_argument('--source', type=str, default='citationv1',
                    help='acmv9, citationv1 or dblpv7')
parser.add_argument('--target', type=str, default='acmv9',
                    help='acmv9, citationv1 or dblpv7')

parser.add_argument('--Weight-domain-dis', type=float, default=1,
                    help="Weight of domain loss")
parser.add_argument('--Weight-node-clf', type=float, default=1,
                    help="Weight of node classification loss")
parser.add_argument('--Weight-edge-clf', type=float, default=1,
                    help="Weight of edge classification loss")
parser.add_argument('--Weight-atten-loss', type=float, default=0.1,
                    help="Weight of supervised attention loss")

parser.add_argument('--gamma', type=float, default=5,
                    help="gamma of Direct Supervision on Graph Attention Learning")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#
 
numRandom = 5 # number of random splits
source = args.source
target = args.target
edge_type = args.edge_type

emb_filename = str(source) + '_' + str(target)
cost_sensitive = False  ##whether give higher weight to scarce class in CE Loss

f = open('./output/' +  emb_filename + '_' + edge_type + '.txt', 'a')
f.write('{}\n'.format(args))
f.flush()


A_s, X_s, Y_s = utils.load_network(str(source) + '.mat')
Y_s = torch.Tensor(Y_s)
num_feats = X_s.shape[1]

features_s = normalize_features(X_s.todense())
features_s = torch.Tensor(features_s)

A_t, X_t, Y_t = utils.load_network(str(target) + '.mat')
Y_t = torch.Tensor(Y_t)
num_nodes_T = X_t.shape[0]
features_t = normalize_features(X_t.todense())
features_t = torch.Tensor(features_t)

##get indices of positive and negative edges

pos_edges_s, neg_edges_s, _, _ = edge_prepare(source)
pos_edges_t, neg_edges_t, _, _ = edge_prepare(target)

labels_pos_s = np.ones((pos_edges_s.shape[0], 1))
labels_neg_s = np.zeros((neg_edges_s.shape[0], 1))
labels_s = np.vstack((labels_pos_s, labels_neg_s))

labels_pos_t = np.ones((pos_edges_t.shape[0], 1))
labels_neg_t = np.zeros((neg_edges_t.shape[0], 1))
labels_t = np.vstack((labels_pos_t, labels_neg_t))


random_state = 0

n_input_s = X_s.shape[1]
n_input_t = X_t.shape[1]

xb_s = torch.FloatTensor(X_s.tocsr().toarray())
xb_t = torch.FloatTensor(X_t.tocsr().toarray())

domain_label_edge = np.vstack([np.tile([1., 0.], [labels_s.shape[0], 1]),
                               np.tile([0., 1.], [labels_t.shape[0], 1])])  # [1,0] for source, [0,1] for target
domain_label_node = np.vstack([np.tile([1., 0.], [X_s.shape[0], 1]), np.tile([0., 1.], [X_t.shape[0], 1])])
best_auc_pr_trp = []
best_auc_roc_trp = []
last_auc_roc_trp = []
last_auc_pr_trp = []
last_AP_neg_trp = []
last_AP_pos_trp = []

g_s = dgl.from_scipy(A_s)
g_s = dgl.remove_self_loop(g_s)
g_s = dgl.add_self_loop(g_s)

g_t = dgl.from_scipy(A_t)
g_t = dgl.remove_self_loop(g_t)
g_t = dgl.add_self_loop(g_t)
if args.gpu < 0:
    cuda = False
else:
    cuda = True
    g_s = g_s.int().to(args.gpu)
    g_t = g_t.int().to(args.gpu)
    Y_t = Y_t.to(args.gpu)
    Y_s = Y_s.to(args.gpu)

heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
while random_state < numRandom:

    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state) if torch.cuda.is_available() else None
    np.random.seed(random_state)
    random.seed(random_state)
    edge_label_s = np.vstack((np.hstack((pos_edges_s, labels_pos_s)), np.hstack((neg_edges_s, labels_neg_s))))
    edge_label_t = np.vstack((np.hstack((pos_edges_t, labels_pos_t)), np.hstack((neg_edges_t, labels_neg_t))))
    edge_label_s_random = np.random.permutation(edge_label_s)
    edge_label_t_random = np.random.permutation(edge_label_t)
    edge_label_s_random_tensor = torch.FloatTensor(edge_label_s_random)
    edge_label_t_random_tensor = torch.FloatTensor(edge_label_t_random)

    random_state = random_state + 1

    print('%d-th random split' % (random_state))

    if edge_type == 'concat':

        model = DGASN(
            num_layers=args.num_layers,
            in_dim=num_feats,
            num_hidden=args.num_hidden,
            heads=heads,
            activation=F.elu,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            dropout=args.dropout,
            input_dim_edge=int((args.num_hidden) * (args.num_heads)) * 2,
            input_dim_node=int((args.num_hidden) * (args.num_heads)),
            hid_dim_edge_clf=args.Hid_dim_edge_clf,
            hid_dim_node_clf=args.Hid_dim_node_clf,
            Hid_dim_edge_domain_1=args.Hid_dim_edge_domain_1,
            Hid_dim_edge_domain_2=args.Hid_dim_edge_domain_2

        )


    else:
        model = DGASN(
            num_layers=args.num_layers,
            in_dim=num_feats,
            num_hidden=args.num_hidden,
            heads=heads,
            activation=F.elu,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            dropout=args.dropout,
            input_dim_edge=int((args.num_hidden) * (args.num_heads)),
            input_dim_node=int((args.num_hidden) * (args.num_heads)),
            hid_dim_edge_clf=args.Hid_dim_edge_clf,
            hid_dim_node_clf=args.Hid_dim_node_clf,
            Hid_dim_edge_domain_1=args.Hid_dim_edge_domain_1,
            Hid_dim_edge_domain_2=args.Hid_dim_edge_domain_2

        )
    if cuda:
        model.cuda()
        features_s = features_s.cuda()
        features_t = features_t.cuda()
        edge_label_s_random_tensor = edge_label_s_random_tensor.cuda()
        edge_label_t_random_tensor = edge_label_t_random_tensor.cuda()


    clf_loss_f = F.binary_cross_entropy_with_logits
    cf_loss_f_node = nn.BCEWithLogitsLoss(reduction='none')
    domain_loss_f = nn.CrossEntropyLoss()


    train_loss_all = []
    train_auc_roc_s_all = []
    test_auc_roc_t_all = []
    domain_loss_edge_all = []
    total_loss_all = []
    AP_s_neg_all = []
    AP_t_neg_all = []

    f1_t = None
    t_total = time.time()
    for cEpoch in range(args.epochs):
        t = time.time()
        p = float(cEpoch) / args.epochs
        lr = args.lr_ini / (1. + 10 * p) ** 0.75
        # lr = args.lr_ini
        grl_lambda = (2. / (1. + np.exp(-10. * p)) - 1)/10 # gradually change from 0 to 0.1
        GradReverse.rate = grl_lambda
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2_w)
        model.train()
        optimizer.zero_grad()
        pred_logit_s, pred_logit_t, pred_logit_node_s, pred_logit_node_t, d_logit, emb_node_s, emb_node_t,  atten_s, _, atten_trp_s, _ = model(
            features_s, features_t, g_s, g_t, edge_label_s_random,
            edge_label_t_random,
            edge_type)
        d_prob = F.softmax(d_logit)
 
        src_s, dst_s = g_s.edges()
        src_s = src_s.type(torch.long)
        dst_s = dst_s.type(torch.long)
 
        index_s1 = edge_label_s_random_tensor[:, 0].type(torch.long)
        index_s2 = edge_label_s_random_tensor[:, 1].type(torch.long)
        atten_adj_s_loss_all = []
        if cuda:
            atten_adj_s = torch.zeros(g_s.num_src_nodes(), g_s.num_dst_nodes()).cuda()
        else:
            atten_adj_s = torch.zeros(g_s.num_src_nodes(), g_s.num_dst_nodes())
        if cuda:
            weit_p = torch.ones((edge_label_s_random_tensor.shape[0])).cuda()
        else:
            weit_p = torch.ones((edge_label_s_random_tensor.shape[0]))
        index_hete_s = np.where(edge_label_s_random[:, 2] == 0)[0]  
        index_homo_s = np.where(edge_label_s_random[:, 2] == 1)[0]
        weit_p[index_hete_s] = args.gamma

 
        atten_loss = 0
        for i in range(0, args.num_layers):
            atten_adj_s[src_s, dst_s] = atten_trp_s[i]
            atten_loss_ij = clf_loss_f(atten_adj_s[index_s1, index_s2].reshape(-1, 1),
                                         edge_label_s_random_tensor[:, 2].reshape(-1, 1), weight=weit_p.reshape(-1, 1))

            atten_loss_ji = clf_loss_f(atten_adj_s[index_s2, index_s1].reshape(-1, 1),
                                         edge_label_s_random_tensor[:, 2].reshape(-1, 1), weight=weit_p.reshape(-1, 1))
            atten_loss_each_layer = (atten_loss_ij + atten_loss_ji) / 2
            atten_loss = atten_loss +atten_loss_each_layer

        if cost_sensitive:  ##give higher weight to scarce class
            clf_loss_edge = clf_loss_f(pred_logit_s, edge_label_s_random_tensor[:, 2].reshape(-1, 1), weight=weit_p)
            clf_loss_node = cf_loss_f_node(pred_logit_node_s, Y_s)
            clf_loss_node = torch.sum(clf_loss_node) / np.sum(Y_s.shape[0])
        else:

            clf_loss_edge = clf_loss_f(pred_logit_s, edge_label_s_random_tensor[:, 2].reshape(-1, 1))
            clf_loss_node = cf_loss_f_node(pred_logit_node_s, Y_s)
            clf_loss_node = torch.sum(clf_loss_node) / np.sum(Y_s.shape[0])

        domain_loss_edge = domain_loss_f(d_logit, torch.argmax(
            torch.FloatTensor(domain_label_edge).to(
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), 1))
        total_loss = clf_loss_edge * args.Weight_edge_clf + clf_loss_node * args.Weight_node_clf \
                     + domain_loss_edge * args.Weight_domain_dis + atten_loss * args.Weight_atten_loss
        total_loss.backward()
        domain_loss_edge_all.append(domain_loss_edge.item())
        total_loss_all.append(total_loss.item())

        optimizer.step()

        '''Compute evaluation on test data by the end of each epoch'''
    
        model.eval()  
        with torch.no_grad():
            GradReverse.rate = 1.0

            pred_logit_xs, pred_logit_xt, pred_logit_node_xs, pred_logit_node_xt, _, emb_node_s, emb_node_t,  atten_xs, atten_xt, atten_trp_xs, atten_trp_xt = model(
                features_s, features_t, g_s, g_t, edge_label_s_random,
                edge_label_t_random, edge_type)
 

        pred_prob_s = F.softmax(pred_logit_xs)
        pred_prob_t = F.softmax(pred_logit_xt)

        auc_roc_train_s = GET_AUC_ROC(edge_label_s_random_tensor[:, 2].reshape(-1, 1), pred_logit_xs)

        auc_roc_test_t = GET_AUC_ROC(edge_label_t_random_tensor[:, 2].reshape(-1, 1), pred_logit_xt)



        AP_s_neg = average_precision_score(1 - edge_label_s_random_tensor[:, 2].reshape(-1, 1).cpu(),
                                           1 - pred_logit_xs.cpu().detach().numpy(), pos_label=1)
        AP_t_neg = average_precision_score(1 - edge_label_t_random_tensor[:, 2].reshape(-1, 1).cpu(),
                                           1 - pred_logit_xt.cpu().detach().numpy(), pos_label=1)

        train_auc_roc_s_all.append(auc_roc_train_s)
        test_auc_roc_t_all.append(auc_roc_test_t)

        AP_s_neg_all.append(AP_s_neg)
        AP_t_neg_all.append(AP_t_neg)

        if (cEpoch + 1) % 10 == 0 or cEpoch == 0:
            print('Epoch: {:04d}'.format(cEpoch + 1),
                  'total_loss: {:.4f}'.format(total_loss.item()),
                  'Source auc_roc: {:.4f}'.format(auc_roc_train_s.item()),
                  'Target testing auc_roc: {:.4f}'.format(auc_roc_test_t.item()),
                  'Source AP_neg: {:.4f}'.format(AP_s_neg.item()),
                  'Target AP_neg: {:.4f}'.format(AP_t_neg.item()),
                  'time: {:.4f}s'.format(time.time() - t))
 
   
    last_auc_roc_test = auc_roc_test_t
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("The last testing auc_roc %f  " % (last_auc_roc_test))


    f.write('%d-th random initialization, the last testing auc_roc %f \n' % (
        random_state, last_auc_roc_test))

    f.write('%d-th random initialization, the last testing AP_neg % f  \n' % (
        random_state,  AP_t_neg))
    f.flush()

    last_auc_roc_trp.append(last_auc_roc_test)
    last_AP_neg_trp.append(AP_t_neg)


last_avg_auc_roc = np.mean(last_auc_roc_trp)
last_std_auc_roc = np.std(last_auc_roc_trp)
last_avg_AP_neg = np.mean(last_AP_neg_trp)
last_std_AP_neg = np.std(last_AP_neg_trp)



print('avg last testing auc_roc over %d random initialization: %f +/- %f' % (
    numRandom, last_avg_auc_roc, last_std_auc_roc))

print(
    'avg last testing AP_neg over %d random initialization: %f +/- %f' % (numRandom, last_avg_AP_neg, last_std_AP_neg))


f.write('avg last testing auc_roc over %d random splits: %f +/- %f  \n' % (
    numRandom, last_avg_auc_roc, last_std_auc_roc))
f.flush()

f.write('avg last testing AP_neg over %d random splits: %f +/- %f  \n' % (
    numRandom, last_avg_AP_neg, last_std_AP_neg))
f.flush()

f.close()