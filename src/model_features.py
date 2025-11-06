"""Models for feature prediction from adjacency matrix.

This module contains GAE and VGAE variants that predict node features from 
the adjacency matrix structure, reversing the typical direction (A -> X instead of X -> A).

Based on:
- DaehanKim (2019) implementation: https://github.com/DaehanKim/vgae_pytorch
- Thomas Kipf Paper: https://arxiv.org/pdf/1611.07308
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import args


class VGAE_Features(nn.Module):
    """Variational Graph Autoencoder for feature prediction.
    
    Encodes adjacency matrix structure into a latent space and decodes to predict
    node features. Uses variational inference with KL divergence regularization.
    """
    
    def __init__(self, adj, num_features):
        """Initialize VGAE for feature prediction.
        
        Args:
            adj: Normalized adjacency matrix (used for graph convolutions)
            num_features: Number of output features per node
        """
        super(VGAE_Features, self).__init__()
        self.num_features = num_features
        
        # Encoder: Start with identity features (node IDs) and apply GCN with adjacency
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        
        # Decoder: Linear layer to reconstruct features from latent representation
        self.feature_decoder = nn.Linear(args.hidden2_dim, num_features)

    def encode(self, X):
        """Encode input through GCN layers to latent distribution.
        
        Args:
            X: Input node features (typically identity matrix for structure-only encoding)
            
        Returns:
            Sampled latent representation Z
        """
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        
        # Sample from latent distribution using reparameterization trick
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def decode(self, Z):
        """Decode latent representation to predict features.
        
        Args:
            Z: Latent node representations
            
        Returns:
            Predicted feature matrix
        """
        X_pred = self.feature_decoder(Z)
        return X_pred

    def forward(self, X):
        """Forward pass: encode then decode.
        
        Args:
            X: Input features (typically identity matrix for adjacency-only input)
            
        Returns:
            Predicted feature matrix
        """
        Z = self.encode(X)
        X_pred = self.decode(Z)
        return X_pred


class GAE_Features(nn.Module):
    """Graph Autoencoder for feature prediction (non-variational).
    
    Similar to VGAE_Features but without the variational component.
    """
    
    def __init__(self, adj, num_features):
        """Initialize GAE for feature prediction.
        
        Args:
            adj: Normalized adjacency matrix (used for graph convolutions)
            num_features: Number of output features per node
        """
        super(GAE_Features, self).__init__()
        self.num_features = num_features
        
        # Encoder: GCN layers
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        
        # Decoder: Linear layer to reconstruct features
        self.feature_decoder = nn.Linear(args.hidden2_dim, num_features)

    def encode(self, X):
        """Encode input through GCN layers.
        
        Args:
            X: Input node features (typically identity matrix)
            
        Returns:
            Latent representation Z
        """
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def decode(self, Z):
        """Decode latent representation to predict features.
        
        Args:
            Z: Latent node representations
            
        Returns:
            Predicted feature matrix
        """
        X_pred = self.feature_decoder(Z)
        return X_pred

    def forward(self, X):
        """Forward pass: encode then decode.
        
        Args:
            X: Input features (typically identity matrix for adjacency-only input)
            
        Returns:
            Predicted feature matrix
        """
        Z = self.encode(X)
        X_pred = self.decode(Z)
        return X_pred


class GraphConvSparse(nn.Module):
    """Sparse graph convolution layer."""
    
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        """Initialize graph convolution layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            adj: Adjacency matrix (normalized)
            activation: Activation function
        """
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim) 
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        """Forward pass through graph convolution.
        
        Args:
            inputs: Input feature matrix
            
        Returns:
            Convolved and activated features
        """
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def glorot_init(input_dim, output_dim):
    """Initialize weights using Glorot/Xavier uniform distribution.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        
    Returns:
        Initialized parameter tensor
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)
