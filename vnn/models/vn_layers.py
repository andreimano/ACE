import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

def create_gen(channels):
        genX=torch.tensor([[0,0,0],[0,0,-1],[0,1,0]])
        genY=torch.tensor([[0,0,1],[0,0,0],[-1,0,0]])
        genZ=torch.tensor([[0,-1,0],[1,0,0],[0,0,0]])

        X_in_list=channels*[genX]
        Y_in_list=channels*[genY]
        Z_in_list=channels*[genZ]
        X_in_bl=torch.block_diag(*X_in_list)
        Y_in_bl=torch.block_diag(*Y_in_list)
        Z_in_bl=torch.block_diag(*Z_in_list)
    
        in_bl=torch.cat([X_in_bl,Y_in_bl,Z_in_bl],dim=0)       
        return in_bl.T

def LieBracketNorm(in_channels,out_channels):

    genIn=create_gen(in_channels).float().to('cuda')
    genOut=create_gen(out_channels).float().to('cuda')
    return genIn.unsqueeze(0).unsqueeze(0),genOut.unsqueeze(0).unsqueeze(0)

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, args=None):
        super(VNLinear, self).__init__()
        self.args = args

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

        self.in_channels=in_channels
        self.out_channels=out_channels
        
    def forward(self, x, disable_equivariance=None):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)

        return x_out
    

class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2, args=None):
        super(VNLinearLeakyReLU, self).__init__()
        self.args = args

        self.dim = dim
        self.negative_slope = negative_slope

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        if args.disable_equivariance:
            self.map_to_feat_nonequiv = nn.Linear(in_channels * 3, out_channels * 3)


        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
            if args.disable_equivariance and args.full_nonequi:
                self.map_to_dir_nonequiv = nn.Linear(in_channels * 3, 3)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
            if args.disable_equivariance and args.full_nonequi:
                self.map_to_dir_nonequiv = nn.Linear(in_channels * 3, out_channels * 3)

        if args.disable_equivariance:
            self.gammas = nn.ParameterList([nn.Parameter(torch.tensor(args.gamma_init_val)) for _ in range(2)])

    def get_nonequi_linear_layers(self):
        return [self.map_to_feat_nonequiv, self.map_to_dir_nonequiv]

    def apply_skip_connection(self, x, x_skip, dense_lin, gamma):
        size = x_skip.size()
        if len(size) == 5:
            batch_size, channels, num_dims, num_points, k = size
            x_skip_flat = x_skip.permute(0, 3, 4, 1, 2).contiguous().view(batch_size, num_points * k, -1)
            x_dense = dense_lin(x_skip_flat)
            out_channels = x.size(1)
            x_dense = x_dense.view(batch_size, num_points, k, out_channels, num_dims).permute(0, 3, 4, 1, 2)
        elif len(size) == 4:
            batch_size, channels, num_dims, num_points = size
            x_skip_flat = x_skip.permute(0, 3, 1, 2).contiguous().view(batch_size, num_points, -1)
            x_dense = dense_lin(x_skip_flat)
            out_channels = x.size(1)
            x_dense = x_dense.view(batch_size, out_channels, num_dims, num_points)
        else:
            raise ValueError(f"Unexpected tensor shape {size}")
        x = x + gamma * x_dense
        return x
    
    def get_gammas(self):
        if self.args.disable_equivariance:
            return [self.gammas[-2], self.gammas[-1]]
        else:
            return []

    def forward(self, x, disable_equivariance):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_skip = x.clone()

        p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)

        if disable_equivariance:
            p = self.apply_skip_connection(p, x_skip, self.map_to_feat_nonequiv, self.gammas[0])

        p = self.batchnorm(p)
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)

        if disable_equivariance and self.args.full_nonequi:
            d = self.apply_skip_connection(d, x_skip, self.map_to_dir_nonequiv, self.gammas[1])

        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out
    
class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2, args=None):
        super(VNLinearLeakyReLU, self).__init__()
        self.args = args
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim, mode=use_batchnorm)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x = self.linear(x)
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        x_out = self.leaky_relu(x)
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature1(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2, args=None):
        super(VNStdFeature, self).__init__()
        self.args = args

        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, args=args)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, args=args)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)

    
    def forward(self, x, disable_equivariance):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)

        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            v1 = z0[:,0,:]
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0



class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2, args=None):
        super(VNStdFeature, self).__init__()
        self.args = args

        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, args=args)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, args=args)

        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)

    def get_gammas(self):
        gammas = self.vn1.get_gammas() + self.vn2.get_gammas()
        return gammas
    
    def get_nonequi_linear_layers(self):
        return self.vn1.get_nonequi_linear_layers() + self.vn2.get_nonequi_linear_layers()

    def forward(self, x, disable_equivariance):
        '''
        x: point features of shape [B, N_feat, 3, N_samples]
        '''

        z0 = x
        z0 = self.vn1(z0, disable_equivariance)
        z0 = self.vn2(z0, disable_equivariance)

        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            v1 = z0[:, 0, :]
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0
