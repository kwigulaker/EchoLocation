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
import open3d as o3d
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
max_nodes = 7000
train = False
test = False

#print("Cuda available: ", torch.cuda.is_available())
#print("Device name:", torch.cuda.get_device_name())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def processData(directory,class_num):
    files = os.listdir(directory)
    # Load dataset in
    for cluster_file in files:
        new_pcd = PCD(directory + "/" + str(cluster_file))
        if(new_pcd.outliers.shape[0] > max_nodes):
            print("PCD over max node size, downsampling...")
            new_pcd.outliers = downSampleRandom(new_pcd.outliers,max_nodes)
        new_pcd.generateGraphNN(neighbor_radius=0.5)
        graphs.append(new_pcd.graph_outliers)
        labels.append(class_num)

def downSampleRandom(pcd,threshold):
    indices = np.random.choice(pcd.shape[0], threshold, replace=False)
    downsampled_pcd = pcd[indices, :]
    return downsampled_pcd

def downSampleVoxel(pcd):
    num_points = np.asarray(pcd.points).shape[0]
    voxel_size = (max_nodes / num_points)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)

    return np.asarray(downsampled_pcd.points)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 32)
        self.lin = torch.nn.Linear(32, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)

class GraphClassificationDataset(Dataset):
    def __init__(self, graphs: List[nx.Graph], labels: List[int]):
        super(GraphClassificationDataset, self).__init__()
        for graph in graphs:
            node_dict = {i: (0,0,0) for i in range(graph.number_of_nodes(),max_nodes)} # Padding to create a uniform graph size
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


def predictOnFiles(dir,model_dir):
    files = os.listdir(dir)
    graphs = []
    pred_labels = []
    model = GCN(in_channels=max_nodes, hidden_channels=1000, out_channels=4)
    model.load_state_dict(torch.load(model_dir,map_location=torch.device('cpu')))

    for cluster_file in files:
        new_pcd = PCD(dir + "/" + str(cluster_file))
        if(new_pcd.outliers.shape[0] > max_nodes):
            print("PCD over max node size, downsampling...")
            new_pcd.outliers = downSampleRandom(new_pcd.outliers,max_nodes)
        new_pcd.generateGraphNN(neighbor_radius=0.5)
        graph = new_pcd.graph_outliers
        node_dict = {i: (0,0,0) for i in range(graph.number_of_nodes(),max_nodes)} # Padding to create a uniform graph size
        graph.add_nodes_from(node_dict)
        graphs.append(graph)

    for graph in graphs:
        edge_index = torch.tensor(list(graph.edges)).t().contiguous()
        x = torch.tensor(to_numpy_matrix(graph), dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        out = model(data)
        pred = out.argmax(dim=1)
        pred_labels.append(pred.item())
    
    ind = 0
    for label in pred_labels:
        np.savetxt(dir + "/" + str(files[ind]) + "_label.txt", np.asarray([label]))
        ind+=1
if(test):
    processData("../EM2040/data/clusters/terrain_xyz",0)
    processData("../EM2040/data/clusters/unknown_xyz",1)
    processData("../EM2040/data/clusters/moorings_xyz",2)
    processData("../EM2040/data/clusters/shipwrecks_xyz",3)

    # Create a PyTorch dataset and dataloader
    dataset = GraphClassificationDataset(graphs, labels)
    dataloader = DataListLoader(dataset, batch_size=1,shuffle=True)

    # Initialize the model and optimizer
    model = GCN(in_channels=max_nodes, hidden_channels=1000, out_channels=4)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.cuda.empty_cache()

    if(train):
        # Train the model
        model.train()
        best_acc = 0.0

        for epoch in range(300):
            epoch_loss = 0.0
            epoch_acc = 0.0
            for data in dataloader:
                data = data[0].to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += (out.argmax(dim=1) == data.y).sum().item()

            epoch_loss /= len(dataset)
            epoch_acc /= len(dataset)
            print(f"Epoch {epoch}, loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.4f}")
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_v2.pt")
    # Test the model
    model.load_state_dict(torch.load('best.pt',map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        num_items = 0
        num_corr = 0
        num_wrong = 0
        for data in dataloader:
            num_items+=1
            data = data[0].to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            #print('Prediction:' +  str(pred) + ", Real: " + str(data.y))
            if(pred == data.y):
                num_corr+=1
            else:
                num_wrong+=1
        print('Number of clouds:' +  str(num_items) + ", Correct predictions: " + str(num_corr) + ", Wrong predictions: " + str(num_wrong))

