#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_kld(z_mean, z_logvar):
    '''
    Calculate KL divergence

    Args:
        z_mean (float): latent mean from encoder
        z_logvar (float): latent logvar from encoder
    
    Returns:
        kld (float): list of SMILE strings for training
    '''

    kld = -0.5 * torch.mean(1 + z_logvar - z_mean**2 - z_logvar.exp())

    return kld 

class ChemVAE(nn.Module):
    def __init__(self):
        super(ChemVAE, self).__init__()

        self.conv_1 = nn.Conv1d(train_oh.shape[1], 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.linear_1 = nn.Linear(30, 435)
        self.linear_2 = nn.Linear(435, 292)
        self.linear_3 = nn.Linear(435, 292)

        self.linear_4 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.linear_5 = nn.Linear(501, train_oh.shape[2])
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_1(x))
        z_mu = self.linear_2(x)
        z_logvar = self.linear_3(x)
        return z_mu, z_logvar

    def sampling(self, mu, log_var):
        '''
        Sample from latent space, z ~ N(μ, σ**2)
        '''
        sigma = torch.exp(log_var / 2)

        #epsilon = torch.randn(sigma.size()).float()
        epsilon = torch.randn_like(sigma).float()

        self.z_mean = mu
        self.z_logvar = log_var

        # use the reparameterization trick
        return mu + sigma * epsilon

    def decode(self, z):
        z = F.selu(self.linear_4(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 60, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_5(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        output = self.decode(z)
        return output