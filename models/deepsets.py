import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# PyTorch Geometric for ModelNet10 dataset and aggregators
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.nn import aggr


# Print available devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
seed = 123
torch.manual_seed(seed)

class DeepSets(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, aggregator='sum', dropout=0.0):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        if aggregator == 'max':
            self.aggregator = aggr.MaxAggregation()
        elif aggregator == 'mean':
            self.aggregator = aggr.MeanAggregation()
        elif aggregator == 'sum':
            self.aggregator = aggr.SumAggregation()
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        h = self.psi(x)
        h = self.aggregator(h, dim=1).squeeze(1)
        y = self.phi(h)

        return y