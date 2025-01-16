import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertModel

from .momu_gnn import MoMuGNN

class MoMu(nn.Module):
    def __init__(self, config):
        super(MoMu, self).__init__()

        self.gin_hidden_dim = config["gin_hidden_dim"]
        self.gin_num_layers = config["gin_num_layers"]
        self.drop_ratio = config["drop_ratio"]
        self.graph_pooling = config["graph_pooling"]
        self.graph_self = config["graph_self"]

        self.bert_dropout = config["bert_dropout"]
        self.bert_hidden_dim = config["bert_hidden_dim"]

        self.projection_dim = config["projection_dim"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_encoder = MoMuGNN(
            num_layer=self.gin_num_layers,
            emb_dim=self.gin_hidden_dim,
            gnn_type='gin',
            drop_ratio=self.drop_ratio,
            JK='last'
        )
        
        state_dict = torch.load(config['pretrained_model_path'], map_location=device)
        self.graph_encoder.load_state_dict(state_dict)

    def encode_structure(self, structure, proj=False):
        h, seq_h, seq_h_mask = self.graph_encoder(structure)
        if proj:
            h = self.graph_proj_head(h)
        return h, seq_h, seq_h_mask

    def encode_structure_with_prob(self, structure, x, atomic_num_list, device):
        h, _ = self.graph_encoder(structure, x, atomic_num_list, device)
        return self.graph_proj_head(h) 