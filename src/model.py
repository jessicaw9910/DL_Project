#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, d_latent, d_length, d_char):
        super(ConvEncoder, self).__init__()
        self.d_latent = d_latent
        self.d_length = d_length
        self.d_char = d_char
        
        self.conv1 = nn.Conv1d(self.d_length, 9, kernel_size=9)
        self.conv2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv3 = nn.Conv1d(9, 10, kernel_size=11)
        
        ## kernel size = 9 + 9 + 11 - 3 = 26
        ## per Gómez-Bombarelli Zinc FC layer dim 196
        self.fc1 = nn.Linear(10 * (self.d_char - 26), 196)
        self.fc_mu = nn.Linear(196, self.d_latent)
        self.fc_logvar = nn.Linear(196, self.d_latent)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        x = F.selu(self.fc1(x))
        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        
        return z_mu, z_logvar

class GRUDecoder(nn.Module):
    def __init__(self, d_latent, d_length, d_char):
        super(GRUDecoder, self).__init__()
        self.d_latent = d_latent
        self.d_length = d_length
        self.d_char = d_char
        
        self.fc1 = nn.Linear(self.d_latent, self.d_latent)
        self.gru = nn.GRU(self.d_latent, 501, 3, batch_first=True)
        self.fc2 = nn.Linear(501, self.d_char)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, z):
        z = F.selu(self.fc1(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.d_length, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.fc2(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

class ChemVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(ChemVAE, self).__init__()
        
        self.encode = encoder
        self.decode = decoder
        
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

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        output = self.decode(z)
        return output