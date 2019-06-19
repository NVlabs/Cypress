##
# @file   weighted_average_wirelength.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Compute weighted-average wirelength according to e-place 
#

import time
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength_cpp as weighted_average_wirelength_cpp
try: 
    import dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength_cuda as weighted_average_wirelength_cuda
    import dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength_cuda_atomic as weighted_average_wirelength_cuda_atomic
    import dreamplace.ops.weighted_average_wirelength.weighted_average_wirelength_cuda_sparse as weighted_average_wirelength_cuda_sparse
except:
    pass 
import pdb 

class WeightedAverageWirelengthFunction(Function):
    """
    @brief compute weighted average wirelength.
    """
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, net_mask, pin_mask, gamma, num_threads):
        """
        @param pos pin location (x array, y array), not cell location 
        @param flat_netpin flat netpin map, length of #pins 
        @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
        @param net_mask whether to compute wirelength, 1 means to compute, 0 means to ignore  
        @param pin_mask whether compute gradient for a pin, 1 means to fill with zero, 0 means to compute
        @param gamma the smaller, the closer to HPWL 
        """
        if pos.is_cuda:
            output = weighted_average_wirelength_cuda.forward(pos.view(pos.numel()), flat_netpin, netpin_start, net_mask, gamma)
        else:
            output = weighted_average_wirelength_cpp.forward(pos.view(pos.numel()), flat_netpin, netpin_start, net_mask, gamma, num_threads)
        ctx.flat_netpin = flat_netpin
        ctx.netpin_start = netpin_start
        ctx.net_mask = net_mask 
        ctx.pin_mask = pin_mask
        ctx.gamma = gamma
        ctx.pos = pos
        ctx.num_threads = num_threads
        return output 

    @staticmethod
    def backward(ctx, grad_pos):
        if grad_pos.is_cuda:
            output = weighted_average_wirelength_cuda.backward(
                    grad_pos, 
                    ctx.pos, 
                    ctx.flat_netpin, 
                    ctx.netpin_start, 
                    ctx.net_mask, 
                    ctx.gamma
                    )
        else:
            output = weighted_average_wirelength_cpp.backward(
                    grad_pos, 
                    ctx.pos, 
                    ctx.flat_netpin, 
                    ctx.netpin_start, 
                    ctx.net_mask, 
                    ctx.gamma, 
                    ctx.num_threads
                    )
        output[:output.numel()//2].masked_fill_(ctx.pin_mask, 0.0)
        output[output.numel()//2:].masked_fill_(ctx.pin_mask, 0.0)
        return output, None, None, None, None, None, None

class WeightedAverageWirelengthAtomicFunction(Function):
    """
    @brief compute weighted average wirelength.
    """
    @staticmethod
    def forward(ctx, pos, pin2net_map, net_mask, pin_mask, gamma):
        """
        @param pos pin location (x array, y array), not cell location 
        @param pin2net_map pin2net map 
        @param net_mask whether to compute wirelength 
        @param pin_mask whether compute gradient for a pin, 1 means to fill with zero, 0 means to compute
        @param gamma the smaller, the closer to HPWL 
        """
        #tt = time.time()
        if pos.is_cuda:
            output = weighted_average_wirelength_cuda_atomic.forward(pos.view(pos.numel()), pin2net_map, net_mask, gamma)
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        ctx.pin2net_map = pin2net_map 
        ctx.net_mask = net_mask 
        ctx.pin_mask = pin_mask
        ctx.gamma = gamma
        ctx.exp_xy = output[1]
        ctx.exp_nxy = output[2]
        ctx.exp_xy_sum = output[3];
        ctx.exp_nxy_sum = output[4];
        ctx.xyexp_xy_sum = output[5];
        ctx.xyexp_nxy_sum = output[6];
        ctx.pos = pos 
        #if torch.isnan(ctx.exp_xy).any() or torch.isnan(ctx.exp_nxy).any() or torch.isnan(ctx.exp_xy_sum).any() or torch.isnan(ctx.exp_nxy_sum).any() or torch.isnan(output[0]).any():
        #    pdb.set_trace()
        torch.cuda.synchronize()
        #print("\t\twirelength forward kernel takes %.3f ms" % ((time.time()-tt)*1000))
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        #tt = time.time()
        if grad_pos.is_cuda:
            output = weighted_average_wirelength_cuda_atomic.backward(
                    grad_pos, 
                    ctx.pos, 
                    ctx.exp_xy.view([-1]), ctx.exp_nxy.view([-1]), 
                    ctx.exp_xy_sum.view([-1]), ctx.exp_nxy_sum.view([-1]), 
                    ctx.xyexp_xy_sum.view([-1]), ctx.xyexp_nxy_sum.view([-1]), 
                    ctx.pin2net_map, 
                    ctx.net_mask, 
                    ctx.gamma
                    )
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        output[:int(output.numel()//2)].masked_fill_(ctx.pin_mask, 0.0)
        output[int(output.numel()//2):].masked_fill_(ctx.pin_mask, 0.0)
        #if torch.isnan(output).any():
        #    pdb.set_trace()
        torch.cuda.synchronize()
        #print("\t\twirelength backward kernel %.3f ms" % ((time.time()-tt)*1000))
        return output, None, None, None, None

class WeightedAverageWirelengthSparseFunction(Function):
    """
    @brief compute weighted average wirelength.
    """
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, netpin_values, pin2net_map, net_mask, pin_mask, gamma):
        """
        @param pos pin location (x array, y array), not cell location 
        @param pin2net_map pin2net map 
        @param net_mask whether to compute wirelength 
        @param pin_mask whether compute gradient for a pin, 1 means to fill with zero, 0 means to compute
        @param gamma the smaller, the closer to HPWL 
        """
        #tt = time.time()
        if pos.is_cuda:
            output = weighted_average_wirelength_cuda_sparse.forward(pos.view(pos.numel()), flat_netpin, netpin_start, netpin_values, pin2net_map, net_mask, gamma)
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        ctx.pin2net_map = pin2net_map 
        ctx.net_mask = net_mask 
        ctx.pin_mask = pin_mask
        ctx.gamma = gamma
        ctx.exp_xy = output[1]
        ctx.exp_nxy = output[2]
        ctx.exp_xy_sum = output[3];
        ctx.exp_nxy_sum = output[4];
        ctx.xyexp_xy_sum = output[5];
        ctx.xyexp_nxy_sum = output[6];
        ctx.pos = pos 
        #if torch.isnan(ctx.exp_xy).any() or torch.isnan(ctx.exp_nxy).any() or torch.isnan(ctx.exp_xy_sum).any() or torch.isnan(ctx.exp_nxy_sum).any() or torch.isnan(output[0]).any():
        #    pdb.set_trace()
        torch.cuda.synchronize()
        #print("\t\twirelength forward kernel takes %.3f ms" % ((time.time()-tt)*1000))
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        #tt = time.time()
        if grad_pos.is_cuda:
            output = weighted_average_wirelength_cuda_sparse.backward(
                    grad_pos, 
                    ctx.pos, 
                    ctx.exp_xy.view([-1]), ctx.exp_nxy.view([-1]), 
                    ctx.exp_xy_sum.view([-1]), ctx.exp_nxy_sum.view([-1]), 
                    ctx.xyexp_xy_sum.view([-1]), ctx.xyexp_nxy_sum.view([-1]), 
                    ctx.pin2net_map, 
                    ctx.net_mask, 
                    ctx.gamma
                    )
        else:
            assert 0, "CPU version NOT IMPLEMENTED"
        output[:output.numel()//2].masked_fill_(ctx.pin_mask, 0.0)
        output[output.numel()//2:].masked_fill_(ctx.pin_mask, 0.0)
        #if torch.isnan(output).any():
        #    pdb.set_trace()
        torch.cuda.synchronize()
        #print("\t\twirelength backward kernel %.3f ms" % ((time.time()-tt)*1000))
        return output, None, None, None, None, None, None, None, None

class WeightedAverageWirelength(nn.Module):
    """ 
    @brief Compute weighted average wirelength. 
    CPU only supports net-by-net algorithm. 
    GPU supports three algorithms: net-by-net, atomic, sparse. 
    Different parameters are required for different algorithms. 
    """
    def __init__(self, flat_netpin=None, netpin_start=None, pin2net_map=None, net_mask=None, pin_mask=None, gamma=None, algorithm='atomic', num_threads=8):
        """
        @brief initialization 
        @param flat_netpin flat netpin map, length of #pins 
        @param netpin_start starting index in netpin map for each net, length of #nets+1, the last entry is #pins  
        @param pin2net_map pin2net map 
        @param net_mask whether to compute wirelength, 1 means to compute, 0 means to ignore  
        @param pin_mask whether compute gradient for a pin, 1 means to fill with zero, 0 means to compute
        @param gamma the smaller, the closer to HPWL 
        @param algorithm must be net-by-net | atomic | sparse 
        """
        super(WeightedAverageWirelength, self).__init__()
        assert net_mask is not None and pin_mask is not None and gamma is not None, "net_mask, pin_mask, gamma are requried parameters"
        if algorithm == 'net-by-net':
            assert flat_netpin is not None and netpin_start is not None, "flat_netpin, netpin_start are requried parameters for algorithm net-by-net"
        elif algorithm == 'atomic':
            assert pin2net_map is not None, "pin2net_map is required for algorithm atomic"
        elif algorithm == 'sparse':
            assert flat_netpin is not None and netpin_start is not None and pin2net_map is not None, "flat_netpin, netpin_start, pin2net_map are requried parameters for algorithm sparse"
        self.flat_netpin = flat_netpin 
        self.netpin_start = netpin_start
        self.netpin_values = None 
        self.pin2net_map = pin2net_map 
        self.net_mask = net_mask 
        self.pin_mask = pin_mask 
        self.gamma = gamma
        self.algorithm = algorithm
        self.num_threads = num_threads
    def forward(self, pos): 
        if pos.is_cuda:
            if self.algorithm == 'net-by-net': 
                return WeightedAverageWirelengthFunction.apply(pos, 
                        self.flat_netpin, 
                        self.netpin_start, 
                        self.net_mask, 
                        self.pin_mask, 
                        self.gamma, 
                        self.num_threads
                        )
            elif self.algorithm == 'atomic':
                return WeightedAverageWirelengthAtomicFunction.apply(pos, 
                        self.pin2net_map, 
                        self.net_mask,
                        self.pin_mask, 
                        self.gamma
                        )
            elif self.algorithm == 'sparse':
                if self.netpin_values is None: 
                    self.netpin_values = torch.ones_like(self.flat_netpin, dtype=pos.dtype)
                return WeightedAverageWirelengthSparseFunction.apply(pos, 
                        self.flat_netpin, 
                        self.netpin_start, 
                        self.netpin_values, 
                        self.pin2net_map, 
                        self.net_mask,
                        self.pin_mask, 
                        self.gamma
                        )
        else: # only net-by-net for CPU 
            return WeightedAverageWirelengthFunction.apply(pos, 
                    self.flat_netpin, 
                    self.netpin_start, 
                    self.net_mask, 
                    self.pin_mask, 
                    self.gamma, 
                    self.num_threads
                    )
