import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_dgcnn_util import get_graph_feature_cross


class STNkd(nn.Module):
    def __init__(self, args, d=64):
        super(STNkd, self).__init__()
        self.args = args
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0, args=args)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0, args=args)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0, args=args)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0, args=args)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0, args=args)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d, args=args)
        self.d = d

        self.conv_list = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.fc1,
            self.fc2,
        ]

    def get_gammas(self):
        gammas = []
        conv_list = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.fc1,
            self.fc2,
        ]

        for c in conv_list:
            gammas += c.get_gammas()

        return gammas

    def forward(self, x, disable_equivariance):
        batchsize = x.size()[0]
        x = self.conv1(x, disable_equivariance=disable_equivariance)
        x = self.conv2(x, disable_equivariance=disable_equivariance)
        x = self.conv3(x, disable_equivariance=disable_equivariance)
        x = self.pool(x)

        x = self.fc1(x, disable_equivariance=disable_equivariance)
        x = self.fc2(x, disable_equivariance=disable_equivariance)
        x = self.fc3(x, disable_equivariance=disable_equivariance)        
        return x
    
    def get_nonequi_linear_layers(self):
        layers = []
        for conv in self.conv_list:
            layers += conv.get_nonequi_linear_layers()

        return layers


class PointNetEncoder(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0, args=args)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0, args=args)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0, args=args)
        
        self.conv3 = VNLinear(128//3, 1024//3, args=args)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0, args=args)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.conv_list = [
            self.conv_pos,
            self.conv1,
            self.conv2,
            self.std_feature
        ]

        if self.feature_transform:
            self.fstn = STNkd(args, d=64//3)
            self.conv_list.append(self.fstn)

    def get_conv_list(self):
        return self.conv_list

    def get_gammas(self):
        gammas = []
        conv_list = [
            self.conv_pos,
            self.conv1,
            self.conv2,
        ]

        for c in conv_list:
            gammas += c.get_gammas()

        gammas += self.std_feature.get_gammas()

        if self.feature_transform:
            gammas += self.fstn.get_gammas()

        return gammas

    def forward(self, x, disable_equivariance):
        B, D, N = x.size()
        

        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat, disable_equivariance=disable_equivariance)

        x = self.pool(x)
        
        x = self.conv1(x, disable_equivariance=disable_equivariance)
        
        if self.feature_transform:
            x_global = self.fstn(x, disable_equivariance=disable_equivariance)
            x_global = x_global.unsqueeze(-1).repeat(1,1,1,N)

            x = torch.cat((x, x_global), 1)
        
        pointfeat = x
        x = self.conv2(x, disable_equivariance=disable_equivariance)
        x = self.conv3(x, disable_equivariance=disable_equivariance)
        x = self.bn3(x)
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x, disable_equivariance=disable_equivariance)
        x = x.view(B, -1, N)
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        trans_feat = None
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
