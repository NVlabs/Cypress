##
# @file   pin_pos.py
# @author Xiaohan Gao
# @date   Sep 2019
# @brief  Compute pin pos
#

import math
import torch
from torch import nn
from torch.autograd import Function

import dreamplace.ops.pin_pos.pin_pos_cpp as pin_pos_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.pin_pos.pin_pos_cuda as pin_pos_cuda
    import dreamplace.ops.pin_pos.pin_pos_cuda_segment as pin_pos_cuda_segment

import pdb


class PinPosFunction(Function):
    """
    @brief Given cell locations, compute pin locations.
    """
    @staticmethod
    def forward(ctx, pos, pin_offset_x, pin_offset_y, theta, pin2node_map,
                flat_node2pin_map, flat_node2pin_start_map,
                num_physical_nodes, h, w):
        ctx.pos = pos.view(pos.numel())
        if pos.is_cuda:
            func = pin_pos_cuda.forward
            output = func(ctx.pos, pin_offset_x, pin_offset_y, theta, pin2node_map,
                      flat_node2pin_map, flat_node2pin_start_map, h, w)
        else:
            func = pin_pos_cpp.forward
            output = func(ctx.pos, pin_offset_x, pin_offset_y, pin2node_map,
                        flat_node2pin_map, flat_node2pin_start_map)
        ctx.pin_offset_x = pin_offset_x
        ctx.pin_offset_y = pin_offset_y
        ctx.pin2node_map = pin2node_map
        ctx.flat_node2pin_map = flat_node2pin_map
        ctx.flat_node2pin_start_map = flat_node2pin_start_map
        ctx.num_physical_nodes = num_physical_nodes
        ctx.theta = theta
        ctx.h = h
        ctx.w = w
        return output

    @staticmethod
    def backward(ctx, grad_pin_pos):
        # grad_pin_pos is not contiguous
        if grad_pin_pos.is_cuda:
            func = pin_pos_cuda.backward
        else:
            func = pin_pos_cpp.backward
        output = func(grad_pin_pos.contiguous(), ctx.pos, ctx.pin_offset_x,
                      ctx.pin_offset_y, ctx.theta, ctx.pin2node_map,
                      ctx.flat_node2pin_map, ctx.flat_node2pin_start_map,
                      ctx.h, ctx.w,
                      ctx.num_physical_nodes)
        # split pos and theta gradients
        num_nodes = ctx.pos.numel() // 2
        grad_pos = output[:num_nodes * 2]
        grad_theta = output[num_nodes * 2:]
        return grad_pos, None, None, grad_theta, None, None, None, None, None, None


class PinPosSegmentFunction(Function):
    """
    @brief Given cell locations, compute pin locations.
    """
    @staticmethod
    def forward(ctx, pos, pin_offset_x, pin_offset_y, pin2node_map,
                flat_node2pin_map, flat_node2pin_start_map,
                num_physical_nodes):
        ctx.pos = pos.view(pos.numel())
        if not pos.is_cuda:
            assert 0, "CPU version NOT implemented"
        else:
            output = pin_pos_cuda_segment.forward(ctx.pos, pin_offset_x,
                                                  pin_offset_y, pin2node_map,
                                                  flat_node2pin_map,
                                                  flat_node2pin_start_map)
        ctx.pin_offset_x = pin_offset_x
        ctx.pin_offset_y = pin_offset_y
        ctx.pin2node_map = pin2node_map
        ctx.flat_node2pin_map = flat_node2pin_map
        ctx.flat_node2pin_start_map = flat_node2pin_start_map
        ctx.num_physical_nodes = num_physical_nodes

        if pos.is_cuda:
            torch.cuda.synchronize()

        return output

    @staticmethod
    def backward(ctx, grad_pin_pos):
        # grad_pin_pos is not contiguous
        if grad_pin_pos.is_cuda:
            output = pin_pos_cuda_segment.backward(
                grad_pin_pos.contiguous(), ctx.pos, ctx.pin_offset_x,
                ctx.pin_offset_y, ctx.pin2node_map, ctx.flat_node2pin_map,
                ctx.flat_node2pin_start_map, ctx.num_physical_nodes)
        else:
            assert 0, "CPU version NOT implemented"
        if grad_pin_pos.is_cuda:
            torch.cuda.synchronize()

        return output, None, None, None, None, None, None


class PinPos(nn.Module):
    """
    @brief Given cell locations, compute pin locations.
    Different from torch.index_add which computes x[index[i]] += t[i], 
    the forward function compute x[i] += t[index[i]]
    """
    def __init__(self,
                 pin_offset_x,
                 pin_offset_y,
                 pin2node_map,
                 flat_node2pin_map,
                 flat_node2pin_start_map,
                 num_physical_nodes,
                 h, w,
                 algorithm='segment',
                 orient_logits=None):
        """
        @brief initialization 
        @param pin_offset pin offset in x or y direction, only computes one direction 
        @param algorithm segment|node-by-node
        """
        super(PinPos, self).__init__()
        self.pin_offset_x = pin_offset_x
        self.pin_offset_y = pin_offset_y
        self.pin2node_map = pin2node_map.long()
        self.flat_node2pin_map = flat_node2pin_map
        self.flat_node2pin_start_map = flat_node2pin_start_map
        self.num_physical_nodes = num_physical_nodes
        self.algorithm = algorithm
        self.orient_logits = orient_logits
        self.theta = None
        self.h = h
        self.w = w

    def forward(self, pos):
        """
        @brief API 
        @param pos cell locations. The array consists of x locations of movable cells, fixed cells, and filler cells, then y locations of them 
        """
        assert pos.numel() % 2 == 0
        num_nodes = pos.numel() // 2
        if self.orient_logits is not None:
            # rotation is enabled
            y = torch.nn.functional.gumbel_softmax(self.orient_logits, tau=1.0, hard=True)
            index_tensor = torch.arange(4).unsqueeze(0).expand_as(y).to(y.device)
            choices = torch.sum(y * index_tensor, dim=1)
            self.theta = choices * math.pi / 2
        else:
            # initialize theta to 0
            self.theta = torch.zeros(num_nodes, dtype=pos.dtype, device=pos.device)

        if pos.is_cuda:
            if self.algorithm == 'segment':
                return PinPosSegmentFunction.apply(
                    pos, self.pin_offset_x, self.pin_offset_y,
                    self.pin2node_map, self.flat_node2pin_map,
                    self.flat_node2pin_start_map, self.num_physical_nodes)
            else:
                return PinPosFunction.apply(pos, self.pin_offset_x,
                                            self.pin_offset_y,
                                            self.theta,
                                            self.pin2node_map,
                                            self.flat_node2pin_map,
                                            self.flat_node2pin_start_map,
                                            self.num_physical_nodes,
                                            self.h, self.w)
        else:
            return PinPosFunction.apply(pos, self.pin_offset_x,
                                        self.pin_offset_y, self.pin2node_map,
                                        self.flat_node2pin_map,
                                        self.flat_node2pin_start_map,
                                        self.num_physical_nodes)
