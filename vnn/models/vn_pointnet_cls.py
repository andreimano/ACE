import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.vn_pointnet import PointNetEncoder

class get_model(nn.Module):
    def __init__(self, args, num_class=40, normal_channel=True, **kwargs):
        super(get_model, self).__init__()
        self.args = args
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(args, global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024//3*6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.conv_list = self.feat.get_conv_list()

        self.gammas = []

        for conv in self.conv_list:
            self.gammas += conv.get_gammas()

        print('Number of gammas:', len(self.gammas)) 

    def forward(self, x, disable_equivariance, disable_equivariance_layerwise=False):
        x_init = torch.flatten(x, start_dim=1)
        x, trans, trans_feat = self.feat(x, disable_equivariance=disable_equivariance_layerwise)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat
    
    def get_nonequi_linear_layers(self):
        layers = []
        for conv in self.conv_list:
            layers += conv.get_nonequi_linear_layers()

        return layers
    
    def get_gammas(self):
        return self.gammas
    
class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        return loss
