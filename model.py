
from torch import nn
import torch
import torch.nn.functional as f
from utils import edge_rep_construct
from flip_gradient import GRL
# from dgl.nn.pytorch import edge_softmax, GATConv
from gat import GATConv

class GAT(nn.Module):
    def __init__(self,

                 num_layers,
                 in_dim,
                 num_hidden,

                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

    def forward(self, inputs, g):
        heads = []
        h = inputs
        atten_trp = []
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h, atten, e_gat = self.gat_layers[l](g, temp, get_attention=True)
            # temp = h.flatten(1)
            atten_trp.append(e_gat.flatten(1).mean(dim=-1))
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
        temp = h.flatten(1)
        return temp, heads, e_gat, atten_trp


class EdgeClassifier(nn.Module):
    def __init__(self, input_dim_edge, hid_dim_edge, dropout):
        super(EdgeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim_edge, hid_dim_edge)
        self.fc2 = nn.Linear(hid_dim_edge, 1)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight)

        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class NodeClassifier(nn.Module):
    def __init__(self, input_dim_mlp_node, hid_dim_node, dropout):
        super(NodeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim_mlp_node, hid_dim_node)
        self.fc2 = nn.Linear(hid_dim_node, 5)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim_edge, dropout, Hid_dim_edge_domain_1, Hid_dim_edge_domain_2):
        super(DomainDiscriminator, self).__init__()
        self.h_dann_1 = nn.Linear(input_dim_edge, Hid_dim_edge_domain_1)
        self.h_dann_2 = nn.Linear(Hid_dim_edge_domain_1, Hid_dim_edge_domain_2)
        self.output_layer = nn.Linear(Hid_dim_edge_domain_2, 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.h_dann_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.h_dann_2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.constant_(self.h_dann_1.bias, 0)
        nn.init.constant_(self.h_dann_2.bias, 0)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, h_grl):
        h_grl = f.relu(self.h_dann_1(h_grl))
        h_grl = f.relu(self.h_dann_2(h_grl))
        d_logit = self.output_layer(h_grl)
        return d_logit


class DGASN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 dropout, input_dim_edge, input_dim_node, hid_dim_node_clf, hid_dim_edge_clf, Hid_dim_edge_domain_1,
                 Hid_dim_edge_domain_2):
        super(DGASN, self).__init__()
        self.network_embedding = GAT(
            num_layers,
            in_dim,
            num_hidden,

            heads,
            activation,
            feat_drop,
            attn_drop,
            negative_slope,
            residual)
        self.edge_classifier = EdgeClassifier(input_dim_edge, hid_dim_edge_clf, dropout)
        self.node_classifier = NodeClassifier(input_dim_node, hid_dim_node_clf, dropout)
        self.domain_discriminator = DomainDiscriminator(input_dim_edge, dropout, Hid_dim_edge_domain_1,
                                                        Hid_dim_edge_domain_2)

        self.grl = GRL()

    def forward(self, features_s, features_t, g_s, g_t, edge_label_s_random, edge_label_t_random, edge_type):

        emb_node_s, heads_s, atten_s, atten_trp_s = self.network_embedding(features_s, g_s)
        emb_node_t, heads_t, atten_t, atten_trp_t = self.network_embedding(features_t, g_t)


        edge_rep_s, edge_rep_s_x0, edge_rep_s_x1 = edge_rep_construct(emb_node_s, edge_label_s_random, edge_type)
        edge_rep_t, edge_rep_t_x0, edge_rep_t_x1 = edge_rep_construct(emb_node_t, edge_label_t_random, edge_type)


        edge_rep = torch.cat((edge_rep_s, edge_rep_t), 0)
        # Node_Classifier
        pred_logit_node_s = self.node_classifier(emb_node_s)
        pred_logit_node_t = self.node_classifier(emb_node_t)
        # Edge_Classifier
        pred_logit_s = self.edge_classifier(edge_rep_s)
        pred_logit_t = self.edge_classifier(edge_rep_t)

        # Domain_Discriminator
        h_grl = self.grl(edge_rep)
        d_logit = self.domain_discriminator(h_grl)

        return pred_logit_s, pred_logit_t, pred_logit_node_s, pred_logit_node_t, d_logit, emb_node_s, emb_node_t,  atten_s, atten_t, atten_trp_s, atten_trp_t