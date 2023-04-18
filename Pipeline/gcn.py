# We want to do entire graph classification --> post convolution we want Z_g = f(sum_of_nodes, adjacency_matrix)
# We have unweighted and undirected graphs.
# Adjacency matrix is therefore binary and symmetric.
# Need to find a node feature matrix H, (assumably nx3), and linear transformation learned matrix W
# Need to make sure the update rule also includes that a central node is connected to itself otherwise it wont be accounted for
# Mean-pooling needs to be done otherwise features will be upscaled to hell
# GCN update rule look into (very popular)

# Dataset:
# terrain: 173 items --> 0
# unknown: 11 items --> 1
# moorings: 20 items --> 2
# shipwrecks: 15 items --> 3

import os
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from pcd_preprocess import PCD
from sklearn.multiclass import OneVsRestClassifier
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataListLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from itertools import chain
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from networkx import to_numpy_matrix
from typing import List
from torch_geometric.data import Dataset, Data

graphs = []
labels = []
def processData(directory,class_num):
    # Load dataset in
    for cluster_file in os.listdir(directory):
        new_pcd = PCD(directory + "/" + str(cluster_file))
        new_pcd.generateGraphNN()
        graphs.append(new_pcd.graph_outliers)
        labels.append(class_num)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data[0].x, data[0].edge_index, data[0].batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)

class GraphClassificationDataset(Dataset):
    def __init__(self, graphs: List[nx.Graph], labels: List[int]):
        super(GraphClassificationDataset, self).__init__()
        for graph in graphs:
            node_dict = {i: (0,0,0) for i in range(graph.number_of_nodes(),40000)} # Padding to create a uniform graph size
            graph.add_nodes_from(node_dict)
        self.graphs = graphs
        self.labels = labels

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        x = torch.tensor(to_numpy_matrix(graph), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        data.y = torch.tensor([label], dtype=torch.long)
        return data


processData("../EM2040/data/clusters/terrain_xyz",0)
processData("../EM2040/data/clusters/unknown_xyz",1)
processData("../EM2040/data/clusters/moorings_xyz",2)
processData("../EM2040/data/clusters/shipwrecks_xyz",3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create a PyTorch dataset and dataloader
dataset = GraphClassificationDataset(graphs, labels)
dataloader = DataListLoader(dataset, batch_size=1,shuffle=True)

# Initialize the model and optimizer
model = GCN(in_channels=40000, hidden_channels=2000, out_channels=4)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
model.train()
for epoch in range(100):
    epoch_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data[0].y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

# Test the model
model.eval()
with torch.no_grad():
    for data in dataloader:
        out = model(data)
        pred = out.argmax(dim=1)
        print('Prediction:', pred)
