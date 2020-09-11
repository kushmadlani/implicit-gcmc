import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import gather_csr, scatter, segment_csr

from inspect import Parameter
from torch import Tensor

# First Layer of the Encoder (implemented by Pytorch Geometric)
# Please the following repository for details.
# https://github.com/rusty1s/pytorch_geometric
class RGCLayer(MessagePassing):
    def __init__(self, config, weight_init):
        super(RGCLayer, self).__init__(aggr='add')
        self.in_c = config.num_nodes
        self.out_c = config.hidden_size[0]
        self.num_users = config.num_users
        self.num_item = config.num_nodes - config.num_users
        self.drop_prob = config.drop_prob
        self.weight_init = weight_init
        self.bn = config.rgc_bn # bool
        self.relu = config.rgc_relu # bool
        self.device = config.device

        self.base_weight = nn.Parameter(torch.Tensor(self.in_c, self.out_c))
        self.relu = nn.ReLU()
        if self.bn:
            self.bn = nn.BatchNorm1d(self.in_c).to(self.device)

        self.reset_parameters(weight_init)

    def reset_parameters(self, weight_init):
        weight_init(self.base_weight, self.in_c, self.out_c)

    def forward(self, x, edge_index, edge_norm):
        return self.propagate(edge_index=edge_index, x=x, edge_norm=edge_norm)

    def propagate(self, edge_index: Tensor, size=None, **kwargs):
        size = self.__check_input__(edge_index, size)

        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)

        out = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        return self.update(out, **update_kwargs)

    def message(self, x_j, edge_norm):
        # collect weight matrix 
        weight = self.base_weight
        weight = self.node_dropout(weight)

        # input vector x_j acts as one hot
        out = weight[x_j].squeeze(dim=1)

        # out is edges x hidden x normalisation
        return out * edge_norm.reshape(-1, 1)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channles]
        if self.bn:
            batchnorm = nn.BatchNorm1d(aggr_out.size(0)).to(self.device)
            aggr_out = batchnorm(aggr_out.unsqueeze(0)).squeeze(0)
        if self.relu:
            aggr_out = self.relu(aggr_out)
        return aggr_out

    def node_dropout(self, weight):
        drop_mask = torch.rand(self.in_c) + (1 - self.drop_prob)
        drop_mask = torch.floor(drop_mask).type(torch.float)
        drop_mask = drop_mask.unsqueeze(1)
        drop_mask = drop_mask.expand(drop_mask.size(0), self.out_c).to(self.device)

        if weight.shape != drop_mask.shape:
            print(weight.shape)
            print(drop_mask.shape)
            
        assert weight.shape == drop_mask.shape
        weight = weight * drop_mask

        return weight

# Second Layer of the Encoder
class DenseLayer(nn.Module):
    def __init__(self, config, weight_init, bias=False):
        super(DenseLayer, self).__init__()
        in_c = config.hidden_size[0]
        out_c = config.hidden_size[1]
        self.device = config.device
        self.weight_init = weight_init

        self.dropout = nn.Dropout(config.drop_prob)
        self.fc = nn.Linear(in_c, out_c, bias=bias)

        self.bn = config.dense_bn
        if self.bn:
            self.bn_u = nn.BatchNorm1d(config.num_users).to(self.device)
            self.bn_i = nn.BatchNorm1d(config.num_nodes - config.num_users).to(self.device)

        self.relu = config.dense_relu
        if self.relu:
            self.relu = nn.ReLU()

    def forward(self, u_features, i_features):
        # user features
        u_features = self.dropout(u_features)
        u_features = self.fc(u_features)
        if self.bn:
            u_features = self.bn_u(
                    u_features.unsqueeze(0)).squeeze()
        if self.relu:
            u_features = self.relu(u_features)

        # item features
        i_features = self.dropout(i_features)
        i_features = self.fc(i_features)
        if self.bn:
            i_features = self.bn_i(
                    i_features.unsqueeze(0)).squeeze()
        if self.relu:
            i_features = self.relu(i_features)

        return u_features, i_features

# Second Layer of the Encoder in presence of Side info
class DenseItemFeatureLayer(nn.Module):
    def __init__(self, config, weight_init, bias=False):
        super(DenseItemFeatureLayer2, self).__init__()
        in_c = config.hidden_size[0]
        out_c = config.hidden_size[1]
        if config.item_side_info:
            n_side_feat = config.n_side_features 
            feat_hidden_dim = config.feat_hidden_size

        self.device = config.device
        self.weight_init = weight_init

        # layers
        self.dropout = nn.Dropout(config.drop_prob)
        self.fc_feat1 = nn.Linear(n_side_feat, feat_hidden_dim, bias=bias)
        self.fc_feat2 = nn.Linear(in_c + feat_hidden_dim, out_c, bias=bias)
        self.fc = nn.Linear(in_c, out_c, bias=bias)
        
        self.bn = config.dense_bn
        if self.bn:
            self.bn_u = nn.BatchNorm1d(config.num_users).to(self.device)
            self.bn_i = nn.BatchNorm1d(config.num_nodes - config.num_users).to(self.device)

        self.relu = config.dense_relu
        if self.relu:
            self.relu = nn.ReLU()

    def forward(self, u_features, i_features, side_info):
        # unpack side_info dict
        i_side_features = side_info['data']

        # user features
        u_features = self.dropout(u_features)
        u_features = self.fc(u_features)
        if self.bn:
            u_features = self.bn_u(
                    u_features.unsqueeze(0)).squeeze()
        if self.relu:
            u_features = self.relu(u_features)

        # dropout item messages
        i_features = self.dropout(i_features)

        # items with side info
        # pass side info through fc layer
        i_side_features = self.fc_feat1(i_side_features.float())

        # concatenate with messages
        i_features = torch.cat((i_features, i_side_features),1)

        # pass through fc layer
        i_features = self.fc_feat2(i_features)
        if self.relu:
            i_features = self.relu(i_features)

        return u_features, i_features


