import torch
import torch.nn as nn
from layers import RGCLayer, DenseLayer, DenseItemFeatureLayer


# Main Model
class GAE(nn.Module):
    def __init__(self, config, weight_init, side_info=None):
        super(GAE, self).__init__()
        self.gcenc = GCEncoder(config, weight_init, side_info=side_info)
        self.bidec = BiDecoder(config, weight_init)

    def forward(self, x, edge_index, edge_norm):
        u_features, i_features = self.gcenc(x, edge_index, edge_norm)
        adj_matrices = self.bidec(u_features, i_features)

        return adj_matrices

# Encoder (will be separated to two layers(RGC and Dense))
class GCEncoder(nn.Module):
    def __init__(self, config, weight_init, side_info=None):
        super(GCEncoder, self).__init__()
        self.num_users = config.num_users
        self.num_items = config.num_items

        self.rgc_layer = RGCLayer(config, weight_init)

        self.side = True if side_info is not None else False
        if self.side:
            self.dense_layer = DenseItemFeatureLayer2(config, weight_init)
            self.side_info = side_info
        else:
            self.dense_layer = DenseLayer(config, weight_init)

    def forward(self, x, edge_index, edge_norm):
        features = self.rgc_layer(x, edge_index, edge_norm)
        u_features, i_features = self.separate_features(features)

        # need to grab correct feature
        if self.side:
            u_features, i_features = self.dense_layer(u_features, i_features, self.side_info)
        else:
            u_features, i_features = self.dense_layer(u_features, i_features)
        return u_features, i_features

    def separate_features(self, features):
        u_features = features[:self.num_users]
        i_features = features[self.num_users:]

        return u_features, i_features


# Decoder
class BiDecoder(nn.Module):
    def __init__(self, config, weight_init):
        super(BiDecoder, self).__init__()

        self.feature_dim = config.hidden_size[1]
        self.apply_drop = config.bidec_drop
        self.dropout = nn.Dropout(config.drop_prob)
        
        self.q_matrix = nn.Parameter(torch.Tensor(self.feature_dim, self.feature_dim))

        self.reset_parameters(weight_init)

    def reset_parameters(self, weight_init):
        nn.init.orthogonal_(self.q_matrix)

    def forward(self, u_features, i_features):
        if self.apply_drop:
            u_features = self.dropout(u_features)
            i_features = self.dropout(i_features)
        
        num_users = u_features.shape[0]
        num_items = i_features.shape[0]
        
        # bilinear form 
        out = torch.chain_matmul(u_features, self.q_matrix, i_features.t()).unsqueeze(-1)

        # pass through sigmoid
        out = torch.sigmoid(out)
        out = out.reshape(num_users * num_items, -1)
        return out   
