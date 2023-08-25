import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
from sklearn.metrics import precision_recall_curve,auc
import sklearn.metrics as sm
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.metrics import f1_score
from warnings import filterwarnings

filterwarnings('ignore')
def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)
    return a, x, y


def load_network_data(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    X[X > 0] = 1  # feature binarization
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    return A, X, Y


def edge_prepare(filename):
    adj, features, labels = load_network_data(filename + '.mat')
    features = features / 1.0
    labels = labels / 1.0
    features_tensor = torch.FloatTensor(features)

    com_labels = np.matmul(labels, labels.transpose())
    com_labels[com_labels > 0] = 1
    com_labels[com_labels == 0] = -1

    adj2 = adj.toarray()
    row, col = np.diag_indices_from(adj2)
    adj2[row, col] = 0.0
    adj = csc_matrix(adj2)

    adj_up = sp.triu(adj, 0)
    com_labels_up = sp.triu(com_labels, 0)
    edge_labels = np.multiply(adj_up.toarray(), com_labels_up.toarray())

    x_pos, y_pos = np.where(edge_labels == 1)
    pos_edges = np.array(list(zip(x_pos, y_pos)))
    x_neg, y_neg = np.where(edge_labels == -1)
    neg_edges = np.array(list(zip(x_neg, y_neg)))

    adj_pos = edge_labels.copy()
    adj_pos = adj_pos + adj_pos.transpose()
    adj_pos[adj_pos == -1] = 0

    adj_pos = csc_matrix(adj_pos)


    return pos_edges,neg_edges,adj,adj_pos



def edge_rep_construct(node_rep, edge_label_random, edge_type):
    if edge_type == 'concat':
        edge_rep = torch.cat((node_rep[edge_label_random[:, 0], :], node_rep[edge_label_random[:, 1], :]), 1)
    elif edge_type == 'L1':
        edge_rep = torch.abs(node_rep[edge_label_random[:, 0], :] - node_rep[edge_label_random[:, 1], :])
    elif edge_type == 'L2':
        edge_rep = torch.square(node_rep[edge_label_random[:, 0], :] - node_rep[edge_label_random[:, 1], :])
    elif edge_type == 'had':
        edge_rep = torch.mul(node_rep[edge_label_random[:, 0], :], node_rep[edge_label_random[:, 1], :])
    elif edge_type == 'avg':
        edge_rep = torch.add(node_rep[edge_label_random[:, 0], :], node_rep[edge_label_random[:, 1], :]) / 2

    return edge_rep,node_rep[edge_label_random[:, 0], :],node_rep[edge_label_random[:, 1], :]

def GET_AUC_ROC(true_label,predict):
    preds = torch.sigmoid(predict)
    preds=preds.cpu().detach().numpy()
    auc_roc=sm.roc_auc_score(true_label.cpu(),preds)
    return auc_roc


def GET_AUC_PR(y_true, y_logits):
    y_prob=torch.sigmoid(y_logits.cpu()).detach().numpy()
    y_prob=1-y_prob
    y_true=1-y_true
    precision, recall, thresholds = precision_recall_curve(y_true.cpu(), y_prob) # calculate precision-recall curve
    AUC_PR = auc(recall, precision)
    return AUC_PR

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1),dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def f1_scores(y_pred, y_true):
    def predict(y_tru, y_pre):
        top_k_list = np.array(np.sum(y_tru, 1), np.int32)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = np.zeros(y_tru.shape[1])
            pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
            prediction.append(np.reshape(pred_i, (1, -1)))
        prediction = np.concatenate(prediction, axis=0)
        return np.array(prediction, np.int32)
    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    return results["micro"], results["macro"]

