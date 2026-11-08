"""Upstream VAE architecture and our small stiffness head; no pretrained loading.

VAE class: Copyright (c) 2023 Li Zheng, MIT License; see LICENSE.
"""
import numpy as np
import torch
from torch import nn
import upstream as up

a_dim, ax_dim, x_dim, latent_dim = 8, 32, 8, 48
adj_vec_dim, ptb_vec_dim, device = 378, 27, 'cpu'
UPPER = np.triu_indices(27)
EDGES = np.triu_indices(27, 1)
PAIRS = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2),(3,3),(4,4),(5,5)]
COMPONENTS = ['C11','C12','C13','C22','C23','C33','C44','C55','C66']


def stiffness_matrix(c):
    shape = (*c.shape[:-1], 6, 6)
    result = c.new_zeros(shape) if torch.is_tensor(c) else np.zeros(shape)
    for k, (i, j) in enumerate(PAIRS):
        result[..., i, j] = result[..., j, i] = c[..., k]
    return result


def hard_geometry(probability, offsets):
    a = np.zeros((27, 27), dtype=np.uint8)
    a[UPPER] = probability >= .5
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0)
    full = np.zeros((27, 3))
    full[up.nodesInit[:, [1, 0, 2]] == .5] = np.clip(offsets, -.5, .5)
    full[abs(full) < .001] = 0
    return a, up.ptb2nodes(full[:, [1, 0, 2]])[:, [1, 0, 2]]


def fresh_model():
    model = vaeModel()
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    for layer in [model.adj_fc_std, model.x_fc_std]:
        nn.init.zeros_(layer.weight)
        nn.init.constant_(layer.bias, -3.)
    return model

class vaeModel(nn.Module):

    def __init__(self):
        super(vaeModel, self).__init__()
        self.en_adj_ch1, self.en_adj_ch2, self.en_adj_ch3, self.en_adj_ch4 = (12, 16, 16, 16)
        self.en_adj_kernel1, self.en_adj_kernel2, self.en_adj_kernel3, self.en_adj_kernel4 = (3, 3, 2, 2)
        self.en_adj_stride1, self.en_adj_stride2, self.en_adj_stride3, self.en_adj_stride4 = (1, 1, 1, 1)
        self.en_adj_dim1, self.en_adj_dim2, self.en_adj_dim3, self.en_adj_dim4 = (512, 512, 512, 128)
        self.en_x_ch1, self.en_x_ch2 = (20, 36)
        self.en_x_kernel1, self.en_x_kernel2 = ((3, 1), (2, 1))
        self.en_x_stride1, self.en_x_stride2 = ((2, 1), 1)
        self.en_x_dim1, self.en_x_dim2 = (12, 1)
        self.en_x_hd1, self.en_x_hd2, self.en_x_hd3, self.en_x_hd4 = (256, 256, 128, latent_dim)
        self.en_adj_fc1 = nn.Linear(adj_vec_dim, self.en_adj_dim1)
        self.en_adj_fc2 = nn.Linear(self.en_adj_dim1, self.en_adj_dim2)
        self.en_adj_fc3 = nn.Linear(self.en_adj_dim2, self.en_adj_dim3)
        self.en_adj_fc4 = nn.Linear(self.en_adj_dim3, self.en_adj_dim4)
        self.en_adj_fc5 = nn.Linear(self.en_adj_dim4, a_dim + ax_dim)
        self.adj_fc_mu = nn.Linear(a_dim + ax_dim, a_dim + ax_dim)
        self.adj_fc_std = nn.Linear(a_dim + ax_dim, a_dim + ax_dim)
        x_hidden_dim = [ptb_vec_dim, 640, 640, 640, 512, ax_dim + x_dim]
        self.en_x_fc1 = nn.Linear(x_hidden_dim[0], x_hidden_dim[1])
        self.en_x_fc2 = nn.Linear(x_hidden_dim[1], x_hidden_dim[2])
        self.en_x_fc3 = nn.Linear(x_hidden_dim[2], x_hidden_dim[3])
        self.en_x_fc4 = nn.Linear(x_hidden_dim[3], x_hidden_dim[4])
        self.en_x_fc5 = nn.Linear(x_hidden_dim[4], x_hidden_dim[5])
        self.x_fc_mu = nn.Linear(ax_dim + x_dim, ax_dim + x_dim)
        self.x_fc_std = nn.Linear(ax_dim + x_dim, ax_dim + x_dim)
        self.adj_de_fc = nn.Linear(24, self.en_x_hd3)
        self.x_de_fc = nn.Linear(24, self.en_x_hd3)
        self.de_adj_fc1 = nn.Linear(a_dim + ax_dim, self.en_adj_dim4)
        self.de_adj_fc2 = nn.Linear(self.en_adj_dim4, self.en_adj_dim3)
        self.de_adj_fc3 = nn.Linear(self.en_adj_dim3, self.en_adj_dim2)
        self.de_adj_fc4 = nn.Linear(self.en_adj_dim2, self.en_adj_dim1)
        self.de_adj_fc5 = nn.Linear(self.en_adj_dim1, adj_vec_dim)
        de_x_hidden_dim = [x_dim + ax_dim, 640, 640, 512, 256, ptb_vec_dim]
        self.de_x_fc1 = nn.Linear(de_x_hidden_dim[0], de_x_hidden_dim[1])
        self.de_x_fc2 = nn.Linear(de_x_hidden_dim[1], de_x_hidden_dim[2])
        self.de_x_fc3 = nn.Linear(de_x_hidden_dim[2], de_x_hidden_dim[3])
        self.de_x_fc4 = nn.Linear(de_x_hidden_dim[3], de_x_hidden_dim[4])
        self.de_x_fc5 = nn.Linear(de_x_hidden_dim[4], de_x_hidden_dim[5])
        self.activation = nn.ReLU()

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(logvar)
        eps = torch.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
        eps = eps.to(device)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps

    def encoder(self, adj, x):
        adj = self.activation(self.en_adj_fc1(adj))
        adj = self.activation(self.en_adj_fc2(adj))
        adj = self.activation(self.en_adj_fc3(adj))
        adj = self.activation(self.en_adj_fc4(adj))
        adj = self.en_adj_fc5(adj)
        x = self.activation(self.en_x_fc1(x))
        x = self.activation(self.en_x_fc2(x))
        x = self.activation(self.en_x_fc3(x))
        x = self.activation(self.en_x_fc4(x))
        x = self.en_x_fc5(x)
        adj_mu = self.adj_fc_mu(adj)
        adj_log_var = self.adj_fc_std(adj)
        adj_z = self.reparameterize(adj_mu, adj_log_var)
        x_mu = self.x_fc_mu(x)
        x_log_var = self.x_fc_std(x)
        x_z = self.reparameterize(x_mu, x_log_var)
        mu_tot = torch.cat((adj_mu, x_mu), dim=-1)
        log_var_tot = torch.cat((adj_log_var, x_log_var), dim=-1)
        mu = torch.zeros([len(mu_tot), a_dim + ax_dim + x_dim]).to(device)
        log_var = torch.zeros([len(mu_tot), a_dim + ax_dim + x_dim]).to(device)
        mu[:, :a_dim] = mu_tot[:, :a_dim]
        mu[:, a_dim:a_dim + ax_dim] = 0.5 * (mu_tot[:, a_dim:a_dim + ax_dim] + mu_tot[:, a_dim + ax_dim:a_dim + 2 * ax_dim])
        mu[:, a_dim + ax_dim:] = mu_tot[:, a_dim + 2 * ax_dim:]
        log_var[:, :a_dim] = log_var_tot[:, :a_dim]
        log_var[:, a_dim:a_dim + ax_dim] = 0.5 * (log_var_tot[:, a_dim:a_dim + ax_dim] + log_var_tot[:, a_dim + ax_dim:a_dim + 2 * ax_dim])
        log_var[:, a_dim + ax_dim:] = log_var_tot[:, a_dim + 2 * ax_dim:]
        z = self.reparameterize(mu, log_var)
        return (z, mu, log_var)

    def decoder(self, z):
        adj_de_input = z[:, :a_dim + ax_dim]
        x_de_input = z[:, a_dim:]
        adj = self.activation(self.de_adj_fc1(adj_de_input))
        adj = self.activation(self.de_adj_fc2(adj))
        adj = self.activation(self.de_adj_fc3(adj))
        adj = self.activation(self.de_adj_fc4(adj))
        adj = self.de_adj_fc5(adj)
        x = self.activation(self.de_x_fc1(x_de_input))
        x = self.activation(self.de_x_fc2(x))
        x = self.activation(self.de_x_fc3(x))
        x = self.activation(self.de_x_fc4(x))
        x = self.de_x_fc5(x)
        adj_out = torch.sigmoid(adj)
        x_out = torch.sigmoid(x) - 0.5
        return (adj_out, x_out)

class PropertyHead(nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(48, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 9))
        self.register_buffer('label_mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('label_std', torch.tensor(std, dtype=torch.float32))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mu):
        return self.label_mean + self.label_std * self.net(mu)
