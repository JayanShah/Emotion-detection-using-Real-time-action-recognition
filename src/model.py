import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. GRAPH DEFINITION (Needed for Model Init) ---
def get_adjacency_matrix():
    # Define the edges based on the training notebook
    edges = [
        (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), 
        (5, 6), (6, 7), (8, 9), (9, 10), (1, 0)
    ]
    # Create identity matrix for 25 joints
    A = torch.eye(25)
    # Add connections (undirected graph)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    return A

# --- 2. MODEL CLASSES ---

class AdaptiveGCN(nn.Module):
    def __init__(self, in_ch, out_ch, adj):
        super().__init__()
        # Initialize Learnable Adjacency with the static graph
        self.PA = nn.Parameter(adj.clone()) 
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (N, C, T, V)
        # 1. Global Context: Learn similarity between all nodes
        global_similarity = torch.matmul(x.mean(dim=2).transpose(1, 2), x.mean(dim=2))
        
        # 2. Add to Learnable Adjacency
        A = self.PA + self.soft(global_similarity)
        
        # 3. Graph Convolution
        x = torch.einsum('nctv,nvw->nctw', x, A)
        return self.conv(x)

class ASTGCN_Block(nn.Module):
    def __init__(self, in_ch, out_ch, adj, stride=1):
        super().__init__()
        self.gcn = AdaptiveGCN(in_ch, out_ch, adj)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # Temporal Conv: Kernel (7,1), Padding (3,0) keeps time dimension same (if stride=1)
            nn.Conv2d(out_ch, out_ch, (7, 1), stride=(stride, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_ch)
        )
        
        # Residual Connection
        if in_ch != out_ch or stride != 1:
            self.res = nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1))
        else:
            self.res = nn.Identity()

    def forward(self, x):
        return F.relu(self.tcn(self.gcn(x)) + self.res(x))

class FinalModel(nn.Module):
    def __init__(self, adj):
        super().__init__()
        # Input has 5 channels: [x, y, vx, vy, dist_chin]
        self.l1 = ASTGCN_Block(5, 64, adj)
        self.l2 = ASTGCN_Block(64, 128, adj, stride=2)
        self.l3 = ASTGCN_Block(128, 256, adj, stride=2)
        self.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(256, 7) # 7 Emotions
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        # Global Average Pooling over Time and Vertices
        return self.fc(x.mean(dim=(2, 3)))