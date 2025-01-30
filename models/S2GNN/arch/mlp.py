import torch
from torch import nn
import pdb 
from .utils import generate_adjacent_matrix, normalized_adj_for_gcn, cosine_similarity_torch, bernstein_approximation, bernstein_approximation_log
from .gcn import GCN
from .bernnet import BernNet
from .chebnetii import ChebNetII
import numpy as np
import torch.nn.functional as F

class ComplexMultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.fc_real = nn.Conv2d(
            in_channels=input_dim, out_channels=output_dim, kernel_size=(1, 1), bias=True)
        self.fc_imag = nn.Conv2d(
            in_channels=input_dim, out_channels=output_dim, kernel_size=(1, 1), bias=True)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """ F_temp(x) + x

        Args:
            input_data (torch.Tensor): input data with shape [B, L, N, D=1]

        Returns:
            torch.Tensor: latent repr
        """

        freq = torch.fft.fftn(input_data, dim=1) # [B, L, N, D=1]
        t_mix = torch.complex(self.fc_real(freq.real) - self.fc_imag(freq.imag), self.fc_real(freq.imag) + self.fc_imag(freq.real))
        return torch.fft.ifftn(t_mix, dim=1).real + input_data

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=False)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=False)
        self.act = nn.ReLU()
        # self.act = nn.GELU()
        self.drop = nn.Dropout(p=0.15)

        # self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = input_data + hidden #* self.resweight                           # residual
        return hidden

class EmbeddingTrainer(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, num_nodes, c_dim, random_project) -> None:
        super().__init__()

        self.node_rp_layer = nn.Linear(num_nodes, c_dim, bias=False)
        self.node_rp_layer.weight.requires_grad = False
        self.node_inv_rp_layer = nn.Linear(c_dim, num_nodes, bias=False)

        self.if_rp = random_project

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        if self.if_rp:
            hidden = self.node_inv_rp_layer(self.node_rp_layer(input_data.transpose(0,1)))
            return hidden.transpose(0,1)

        else:
            return input_data

class NFConnection(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, n_kernel) -> None:
        super().__init__()
      
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=n_kernel, kernel_size=(1, input_dim), bias=False)

        # 全一初始化
        self.conv1.weight = nn.Parameter(torch.ones(n_kernel, 1, 1, input_dim), requires_grad=True)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """
        hidden = self.conv1(input_data.transpose(1,3))

        return hidden, self.conv1.weight

class SpatiaEncoder(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, hidden_dim, output_dim, num_gnn_layer, beta_init, gnn_type) -> None:
        super().__init__()
    
        self.mlp = MultiLayerPerceptron(hidden_dim, hidden_dim)
        self.gnn_type = gnn_type
        if gnn_type == 'gcn':
            self.gnn = GCN(hidden_dim, num_layer=num_gnn_layer, init=beta_init)

        elif gnn_type == 'bernnet':
            self.gnn = BernNet(num_gnn_layer)

        elif gnn_type == 'chebnetii':
            self.gnn = ChebNetII(num_gnn_layer)

        self.projection = nn.Conv2d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=(1, 1), bias=False)
        # self.alpha = nn.Parameter(torch.ones(3, 1, 1), requires_grad=True)


    def forward(self, input_data: torch.Tensor, node_embedding: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """ 
        # _, E, N = adj.shape
        B, d, _, _ = input_data.shape
        hidden = self.mlp(input_data) 

        adp_adj = cosine_similarity_torch(node_embedding).relu()

        # adj_mat = self.alpha.softmax(0) * adj_mat

        hidden_in = hidden.transpose(1,2).squeeze(-1)
        if self.gnn_type == 'gcn':
            adj_mat = generate_adjacent_matrix(adp_adj)
            hidden_out, betas = self.gnn(hidden_in, adj_mat.sum(0, keepdim=True).expand(B, -1, -1))

        elif self.gnn_type == 'bernnet':
            hidden_out, betas = self.gnn(hidden_in, adp_adj)

        elif self.gnn_type == 'chebnetii':
            hidden_out, betas = self.gnn(hidden_in, adp_adj)

        else:
            hidden_out = hidden_in
            betas = None

        hidden_out = hidden_out.transpose(1,2).unsqueeze(-1)
        hidden_out = self.projection(hidden_out)
        return hidden_out, betas, adp_adj
