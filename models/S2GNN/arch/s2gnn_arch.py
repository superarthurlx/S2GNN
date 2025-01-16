import torch
from torch import nn
import numpy as np 
import pdb

from .mlp import MultiLayerPerceptron, ComplexMultiLayerPerceptron, SpatiaEncoder, NFConnection, EmbeddingTrainer
from .utils import link_to_onehot

class S2GNN(nn.Module):

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.embed_sparsity = model_args["embed_sparsity"]
        self.embed_std = model_args["embed_std"]
        self.constant_c = model_args["constant"]

        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.num_gnn_layer = model_args["num_gnn_layer"]
        self.beta_init = model_args["init"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_node = model_args["if_node"]

        self.topk = model_args["topk"]
        self.n_kernel = model_args['n_kernel']
        self.if_feat = model_args['if_feat']
        self.use_bern = model_args['use_bern']
        self.stop_grad = model_args['stop_grad']
        self.if_random_project = model_args["if_rp"]
        self.gnn_type = model_args['gnn_type']


        # node embeddings
        if self.if_node:
            if self.if_random_project:
                self.node_emb = self.init_emb(self.num_nodes, grad=False)
                h = self.constant_c * self.embed_sparsity * np.log(self.num_nodes / self.embed_sparsity)
            
            else:
                h = self.num_nodes # whatever
                self.node_emb = self.init_emb(self.num_nodes, grad=True)

            self.node_emb_trainer = EmbeddingTrainer(self.num_nodes, int(h), self.if_random_project)

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = self.init_emb(self.time_of_day_size, grad=True)

        if self.if_day_in_week:
            self.day_in_week_emb = self.init_emb(self.day_of_week_size, grad=True)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=False)

        self.n_feature = int(1 + self.if_time_in_day + self.if_day_in_week + self.if_node)
        # self.spatial_hidden_dim = (1 + self.n_feature) * self.embed_dim

        if self.if_feat:
            self.feature_emb = self.init_emb(self.n_kernel)
            self.temporal_hidden_dim = (self.n_feature + self.topk) * self.embed_dim
            self.spatial_hidden_dim = (1 + self.n_feature) * self.embed_dim

            self.node_feature_connection = NFConnection(self.input_len, self.n_kernel, self.use_bern) 

        else:
            self.temporal_hidden_dim = (1 + self.n_feature) * self.embed_dim
            self.spatial_hidden_dim = self.n_feature * self.embed_dim
        # self.temporal_hidden_dim = self.n_feature * self.embed_dim
        

        
        self.spatial_encoder = SpatiaEncoder(self.spatial_hidden_dim, self.embed_dim, self.num_gnn_layer, self.beta_init, self.gnn_type)
        self.temporal_encoder = nn.Sequential(*[MultiLayerPerceptron(self.temporal_hidden_dim, self.temporal_hidden_dim) for _ in range(self.num_layer)])
        # self.temporal_encoder = nn.Sequential(*[ComplexMultiLayerPerceptron(self.input_len, self.input_len) for _ in range(self.num_layer)])

        # regression
        # self.regression_layer1 = nn.Conv2d(in_channels=self.temporal_hidden_dim, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        # self.regression_layer2 = nn.Conv2d(in_channels=self.embed_dim*self.num_patch, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.regression_layer = nn.Conv2d(in_channels=self.temporal_hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def init_emb(self, granularity: int, embed_dim = None, grad: bool = True):
        if not embed_dim:
            emb = nn.Parameter(torch.empty(granularity, self.embed_dim), requires_grad=grad)
        else:
            emb = nn.Parameter(torch.empty(granularity, embed_dim), requires_grad=grad)
        nn.init.sparse_(emb, sparsity=self.embed_sparsity, std=self.embed_std)
        # nn.init.xavier_uniform_(emb)
        return emb

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]
      
        if self.if_time_in_day:

            t_i_d_data = history_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[(d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)

        time_series_emb = self.time_series_emb_layer(input_data)

        if train and epoch > self.stop_grad:
            self.node_emb_trainer.node_inv_rp_layer.weight.requires_grad = False
            self.node_emb.requires_grad = False

        self.projected_node_emb = self.node_emb_trainer(self.node_emb)

        node_emb = []
        if self.if_node:
            # expand node embeddings
            node_emb.append(self.projected_node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        temporal_enc_in = [torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)]
        spatial_enc_in = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        if self.if_feat:

            connection, feature_sequence = self.node_feature_connection(input_data)
            c_weight, c_index = torch.topk(connection, dim=1, k=self.topk) # (B, k, N, 1)
            c_weight = c_weight.squeeze(-1).transpose(1,2) # (B, N, k)
            c_index = c_index.squeeze(-1).transpose(1,2) # (B, N, k)

            for i in range(self.topk):
                feature_input = c_weight[..., i].sigmoid().unsqueeze(-1) * self.feature_emb[c_index[..., i].type(torch.LongTensor)] 
                feature_input = feature_input.transpose(1,2).unsqueeze(-1)
                s_enc_in = torch.cat([spatial_enc_in, feature_input], dim=1) # (B, d, N, 1)
                s_enc_out, betas, self.adp = self.spatial_encoder(s_enc_in, self.projected_node_emb)
                temporal_enc_in.append(s_enc_out)

        else:

            s_enc_in = spatial_enc_in # (B, d, N, 1)
            s_enc_out, betas, self.adp = self.spatial_encoder(s_enc_in, self.projected_node_emb)
            temporal_enc_in.append(s_enc_out)

        temporal_enc_in = torch.cat(temporal_enc_in, dim=1) # (B, d, N, 1)
        temporal_enc_out = self.temporal_encoder(temporal_enc_in)
        
        prediction = self.regression_layer(temporal_enc_out)
        # print(betas)
        return {"prediction":prediction}



