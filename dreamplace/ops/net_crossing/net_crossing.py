##
# @file   net_crossing.py
# @author Niansong Zhang
# @date   Jul 2024
# @brief  Compute net crossing
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.net_crossing.net_crossing_cpp as net_crossing_cpp
import dreamplace.configure as configure
# if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
#     import dreamplace.ops.density_map.density_map_cuda as density_map_cuda

import pdb

class NetCrossingFunction(Function):
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, net_mask, _lambda, _mu, _sigma):
        if pos.is_cuda:
            assert 0, "CUDA version NOT IMPLEMENTED"
        else:
            output = net_crossing_cpp.forward(pos.view(pos.numel()), flat_netpin, netpin_start, net_mask, _lambda, _mu, _sigma)

        ctx.net_mask = net_mask
        ctx.grad_intermediate = output[1]
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        if grad_pos.is_cuda:
            assert 0, "CUDA version NOT IMPLEMENTED"
        else:
            output = net_crossing_cpp.backward(grad_pos, ctx.grad_intermediate)

        return output

class NetCrossing(nn.Module):

    def __init__(self,
                flat_netpin=None,
                netpin_start=None,
                net_mask=None,
                _lambda=None,
                _mu=None,
                _sigma=None):
        super(NetCrossing, self).__init__()
        assert net_mask is not None \
            and _lambda is not None \
            and _mu is not None \
            and _sigma is not None, \
            "net_mask, _lambda, _mu, _sigma cannot be None"
        
        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.net_mask = net_mask
        self._lambda = _lambda
        self._mu = _mu
        self._sigma = _sigma

    def forward(self, pos):
        if pos.is_cuda:
            assert 0, "CUDA version NOT IMPLEMENTED"
        else:
            return NetCrossingFunction.apply(
                pos, self.flat_netpin, self.netpin_start, self.net_mask, self._lambda, self._mu, self._sigma)