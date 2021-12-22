import torch
import torch.nn.functional as F
from torch import nn

# # define the network class
# class SimpleNet(nn.Module):
#     def __init__(self, in_dim):
#         # call constructor from superclass
#         super().__init__()
        
#         # define network layers
#         self.fc1 = nn.Linear(in_dim, 100)
#         self.fc2 = nn.Linear(100, 10)
#         self.fc3 = nn.Linear(10, 1)

#     def forward(self, x):
#         # define forward pass
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x


# define the network class
class SimpleNet(nn.Module):
    def __init__(self, in_dim, mid_num=3):
        # call constructor from superclass
        super().__init__()
        self.in_dim=in_dim
        self.mid_num = mid_num
        
        # define network layers
        self.fc1 = nn.Linear(in_dim, 100)
        self.fc_mid=[]
        self.fc_mid = nn.ModuleList([nn.Linear(100, 100) for i in range(self.mid_num)])
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        for i in range(self.mid_num):
            x=F.relu(self.fc_mid[i](x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# class SimpleNet(nn.Module):
#     def __init__(self, in_dim):
#         # call constructor from superclass
#         super().__init__()
        
#         # define network layers
#         self.fc1 = nn.Linear(in_dim, 100)
#         self.fc3 = nn.Linear(100, 1)

#     def forward(self, x):
#         # define forward pass
#         x = F.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x