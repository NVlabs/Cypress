##
# @file   electric_potential.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  electric potential according to e-place (http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf)
#

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import math
import numpy as np
import time
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
import logging

import dreamplace.ops.dct.discrete_spectral_transform as discrete_spectral_transform

import dreamplace.ops.dct.dct2_fft2 as dct
from dreamplace.ops.dct.discrete_spectral_transform import get_exact_expk as precompute_expk

#import dreamplace.ops.dct.dct as dct
#from dreamplace.ops.dct.discrete_spectral_transform import get_expk as precompute_expk

from dreamplace.ops.electric_potential.electric_overflow import ElectricDensityMapFunction as ElectricDensityMapFunction
from dreamplace.ops.electric_potential.electric_overflow import ElectricOverflow as ElectricOverflow

import dreamplace.ops.electric_potential.electric_potential_cpp as electric_potential_cpp
import dreamplace.configure as configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    import dreamplace.ops.electric_potential.electric_potential_cuda as electric_potential_cuda

import pdb
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# global variable for plot
plot_count = 0


class ElectricPotentialFunction(Function):
    """
    @brief compute electric potential according to e-place.
    """
    @staticmethod
    def forward(
        ctx,
        pos,
        node_size_x_clamped,
        node_size_y_clamped,
        offset_x,
        offset_y,
        ratio,
        bin_center_x,
        bin_center_y,
        initial_density_map,
        target_density,
        xl,
        yl,
        xh,
        yh,
        bin_size_x,
        bin_size_y,
        num_movable_nodes,
        num_filler_nodes,
        padding,
        padding_mask,  # same dimensions as density map, with padding regions to be 1
        num_bins_x,
        num_bins_y,
        num_movable_impacted_bins_x,
        num_movable_impacted_bins_y,
        num_filler_impacted_bins_x,
        num_filler_impacted_bins_y,
        deterministic_flag,
        sorted_node_map,
        exact_expkM=None,  # exp(-j*pi*k/M)
        exact_expkN=None,  # exp(-j*pi*k/N)
        inv_wu2_plus_wv2=None,  # 1.0/(wu^2 + wv^2)
        wu_by_wu2_plus_wv2_half=None,  # wu/(wu^2 + wv^2)/2
        wv_by_wu2_plus_wv2_half=None,  # wv/(wu^2 + wv^2)/2
        dct2=None,
        idct2=None,
        idct_idxst=None,
        idxst_idct=None,
        fast_mode=True,  # fast mode will discard some computation
        fence_region_map=None,
        fence_region_mask=None,
        filler_beg=None,
        filler_end=None
    ):

        tt = time.time()

        # if(fence_region_map is not None):
        #     added_target_density = 1.3 * target_density.data.item()
        #     added_density = (fence_region_map>fence_region_map.min().data.item()).float().sum()*added_target_density
        #     ori_init_density = initial_density_map.sum() #/ (bin_size_x * bin_size_y)
        #     print(added_density, ori_init_density)
        #     initial_density_map = initial_density_map*ori_init_density/(added_density+ori_init_density)

        if(fence_region_map is not None):
            # print("fence_region_map std:", fence_region_map.std(), fence_region_map.max(), fence_region_map.min())

            # initial_density_mask = initial_density_map > initial_density_map.min().data.item()
            # fence_region_density_mask = fence_region_map > fence_region_map.min().data.item()
            # masked_ratio = (fence_region_mask).float().mean().item()
            # if(masked_ratio > 0.9):
            #     # fence_region_factor = 0.65#(1.2 + (1.4-1.2)*(fence_region_density_mask).float().mean())
            #     fence_region_factor = 0.41#(1.2 + (1.4-1.2)*(fence_region_density_mask).float().mean())
            #     fence_region_factor = 0.7#(1.2 + (1.4-1.2)*(fence_region_density_mask).float().mean())
            #     # fence_region_factor = 0.5+(1-0.5) * 10*(masked_ratio-0.9)#(1.2 + (1.4-1.2)*(fence_region_density_mask).float().mean())
            #     # print("decrease density factor", masked_ratio, fence_region_factor)
            # else:
            #     fence_region_factor = 1
            # initial_density_factor = 1#(1.2 + (1.4-1.2)*(initial_density_mask).float().mean())
            ### mask all other cells to avoid interference
            node_size_x_clamped = node_size_x_clamped.clone()
            node_size_x_clamped[:num_movable_nodes].masked_fill_(fence_region_mask, 0)
            node_size_y_clamped[:num_movable_nodes].masked_fill_(fence_region_mask, 0)
            ### remove the influence of unseen fillers from density map
            ### don't need to handle this in backward as movement of zero-size cells has no influence
            if(num_filler_nodes > 0 and filler_beg is not None):
                num_nodes = node_size_x_clamped.size(0)
                node_size_x_clamped = node_size_x_clamped.clone()
                node_size_x_clamped[num_nodes-num_filler_nodes:num_nodes-num_filler_nodes + filler_beg].zero_()
                node_size_x_clamped[num_nodes-num_filler_nodes + filler_end:].zero_()
                node_size_y_clamped = node_size_y_clamped.clone()
                node_size_y_clamped[num_nodes-num_filler_nodes:num_nodes-num_filler_nodes + filler_beg].zero_()
                node_size_y_clamped[num_nodes-num_filler_nodes + filler_end:].zero_()

        density_map = ElectricDensityMapFunction.forward(
            pos, node_size_x_clamped, node_size_y_clamped, offset_x, offset_y,
            ratio, bin_center_x, bin_center_y, initial_density_map,
            target_density, xl, yl, xh, yh, bin_size_x, bin_size_y,
            num_movable_nodes, num_filler_nodes, padding, padding_mask,
            num_bins_x, num_bins_y, num_movable_impacted_bins_x,
            num_movable_impacted_bins_y, num_filler_impacted_bins_x,
            num_filler_impacted_bins_y, deterministic_flag, sorted_node_map)

        # output consists of (density_cost, density_map, max_density)
        ctx.node_size_x_clamped = node_size_x_clamped
        ctx.node_size_y_clamped = node_size_y_clamped
        ctx.offset_x = offset_x
        ctx.offset_y = offset_y
        ctx.ratio = ratio
        ctx.bin_center_x = bin_center_x
        ctx.bin_center_y = bin_center_y
        ctx.target_density = target_density
        ctx.xl = xl
        ctx.yl = yl
        ctx.xh = xh
        ctx.yh = yh
        ctx.bin_size_x = bin_size_x
        ctx.bin_size_y = bin_size_y
        ctx.num_movable_nodes = num_movable_nodes
        ctx.num_filler_nodes = num_filler_nodes
        ctx.padding = padding
        ctx.num_bins_x = num_bins_x
        ctx.num_bins_y = num_bins_y
        ctx.num_movable_impacted_bins_x = num_movable_impacted_bins_x
        ctx.num_movable_impacted_bins_y = num_movable_impacted_bins_y
        ctx.num_filler_impacted_bins_x = num_filler_impacted_bins_x
        ctx.num_filler_impacted_bins_y = num_filler_impacted_bins_y
        ctx.pos = pos
        ctx.sorted_node_map = sorted_node_map
        #density_map = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        #ctx.field_map_x = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        #ctx.field_map_y = torch.ones([ctx.num_bins_x, ctx.num_bins_y], dtype=pos.dtype, device=pos.device)
        # return torch.zeros(1, dtype=pos.dtype, device=pos.device)

        # for DCT
        M = num_bins_x
        N = num_bins_y

        # wu and wv
        if inv_wu2_plus_wv2 is None:
            wu = torch.arange(M,
                              dtype=density_map.dtype,
                              device=density_map.device).mul(2 * np.pi /
                                                             M).view([M, 1])
            wv = torch.arange(N,
                              dtype=density_map.dtype,
                              device=density_map.device).mul(2 * np.pi /
                                                             N).view([1, N])
            wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
            wu2_plus_wv2[0,
                         0] = 1.0  # avoid zero-division, it will be zeroed out
            inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
            inv_wu2_plus_wv2[0, 0] = 0.0
            wu_by_wu2_plus_wv2_half = wu.mul(inv_wu2_plus_wv2).mul_(1. / 2)
            wv_by_wu2_plus_wv2_half = wv.mul(inv_wu2_plus_wv2).mul_(1. / 2)

        # compute auv
        # if(fence_region_map is not None):
        #     initial_density_map = fence_region_factor * (initial_density_map).masked_scatter(fence_region_density_mask,  fence_region_map[fence_region_density_mask])
        # density_map.add_(initial_density_map)
        density_map.mul_(1.0 / (ctx.bin_size_x * ctx.bin_size_y))


        #auv = discrete_spectral_transform.dct2_2N(density_map, expk0=exact_expkM, expk1=exact_expkN)
        auv = dct2.forward(density_map)

        # compute field xi
        auv_by_wu2_plus_wv2_wu = auv.mul(wu_by_wu2_plus_wv2_half)
        auv_by_wu2_plus_wv2_wv = auv.mul(wv_by_wu2_plus_wv2_half)

        #ctx.field_map_x = discrete_spectral_transform.idsct2(auv_by_wu2_plus_wv2_wu, exact_expkM, exact_expkN).contiguous()
        ctx.field_map_x = idxst_idct.forward(auv_by_wu2_plus_wv2_wu)
        #ctx.field_map_y = discrete_spectral_transform.idcst2(auv_by_wu2_plus_wv2_wv, exact_expkM, exact_expkN).contiguous()
        ctx.field_map_y = idct_idxst.forward(auv_by_wu2_plus_wv2_wv)

        # energy = \sum q*phi
        # it takes around 80% of the computation time
        # so I will not always evaluate it
        if fast_mode:  # dummy for invoking backward propagation
            energy = torch.zeros(1, dtype=pos.dtype, device=pos.device)
        else:
            # compute potential phi
            # auv / (wu**2 + wv**2)
            auv_by_wu2_plus_wv2 = auv.mul(inv_wu2_plus_wv2)
            #potential_map = discrete_spectral_transform.idcct2(auv_by_wu2_plus_wv2, exact_expkM, exact_expkN)
            potential_map = idct2.forward(auv_by_wu2_plus_wv2)
            # compute energy
            energy = potential_map.mul(density_map).sum()

        # torch.set_printoptions(precision=10)
        # logger.debug("initial_density_map")
        # logger.debug(initial_density_map/(ctx.bin_size_x*ctx.bin_size_y))
        # logger.debug("density_map")
        # logger.debug(density_map/(ctx.bin_size_x*ctx.bin_size_y))
        # logger.debug("auv_by_wu2_plus_wv2")
        # logger.debug(auv_by_wu2_plus_wv2)
        # logger.debug("potential_map")
        # logger.debug(potential_map)
        # logger.debug("field_map_x")
        # logger.debug(ctx.field_map_x)
        # logger.debug("field_map_y")
        # logger.debug(ctx.field_map_y)

        #global plot_count
        # if plot_count >= 600 and plot_count % 1 == 0:
        #    logger.debug("density_map")
        #    plot(plot_count, density_map.clone().div(bin_size_x*bin_size_y).cpu().numpy(), padding, "summary/%d.density_map" % (plot_count))
        #    logger.debug("potential_map")
        #    plot(plot_count, potential_map.clone().cpu().numpy(), padding, "summary/%d.potential_map" % (plot_count))
        #    logger.debug("field_map_x")
        #    plot(plot_count, ctx.field_map_x.clone().cpu().numpy(), padding, "summary/%d.field_map_x" % (plot_count))
        #    logger.debug("field_map_y")
        #    plot(plot_count, ctx.field_map_y.clone().cpu().numpy(), padding, "summary/%d.field_map_y" % (plot_count))
        #plot_count += 1

        if pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("density forward %.3f ms" % ((time.time() - tt) * 1000))
        return energy

    @staticmethod
    def backward(ctx, grad_pos):
        tt = time.time()
        if grad_pos.is_cuda:
            output = -electric_potential_cuda.electric_force(
                grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_movable_impacted_bins_x,
                ctx.num_movable_impacted_bins_y,
                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                ctx.field_map_x.view([-1]), ctx.field_map_y.view(
                    [-1]), ctx.pos, ctx.node_size_x_clamped,
                ctx.node_size_y_clamped, ctx.offset_x, ctx.offset_y, ctx.ratio,
                ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh,
                ctx.yh, ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes,
                ctx.num_filler_nodes, ctx.sorted_node_map)
        else:
            output = -electric_potential_cpp.electric_force(
                grad_pos, ctx.num_bins_x, ctx.num_bins_y,
                ctx.num_movable_impacted_bins_x,
                ctx.num_movable_impacted_bins_y,
                ctx.num_filler_impacted_bins_x, ctx.num_filler_impacted_bins_y,
                ctx.field_map_x.view([-1]), ctx.field_map_y.view(
                    [-1]), ctx.pos, ctx.node_size_x_clamped,
                ctx.node_size_y_clamped, ctx.offset_x, ctx.offset_y, ctx.ratio,
                ctx.bin_center_x, ctx.bin_center_y, ctx.xl, ctx.yl, ctx.xh,
                ctx.yh, ctx.bin_size_x, ctx.bin_size_y, ctx.num_movable_nodes,
                ctx.num_filler_nodes)

        #global plot_count
        # if plot_count >= 300:
        #    indices = (ctx.pos[ctx.pos.numel()/2-ctx.num_filler_nodes:ctx.pos.numel()/2] < ctx.xl+ctx.bin_size_x).nonzero()
        #    pdb.set_trace()

        #pgradx = []
        #pgrady = []
        # with open("/home/polaris/yibolin/Libraries/RePlAce/output/ispd/adaptec1.eplace/gradient.csv", "r") as f:
        #    for line in f:
        #        tokens = line.strip().split(" ")
        #        pgradx.append(float(tokens[3].strip()))
        #        pgrady.append(float(tokens[4].strip()))
        #pgrad = np.concatenate([np.array(pgradx), np.array(pgrady)])

        #output = torch.empty_like(ctx.pos).uniform_(0.0, 0.1)
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("density backward %.3f ms" % ((time.time() - tt) * 1000))
        return output, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None


class ElectricPotential(ElectricOverflow):
    """
    @brief Compute electric potential according to e-place
    """
    def __init__(
        self,
        node_size_x,
        node_size_y,
        bin_center_x,
        bin_center_y,
        target_density,
        xl,
        yl,
        xh,
        yh,
        bin_size_x,
        bin_size_y,
        num_movable_nodes,
        num_terminals,
        num_filler_nodes,
        padding,
        deterministic_flag,  # control whether to use deterministic routine
        sorted_node_map,
        movable_macro_mask=None,
        fast_mode=False,
        fence_regions=None, # [n_subregion, 4] as dummy macros added to initial density. (xl,yl,xh,yh) rectangles
        fence_region_mask=None,
        placedb=None
        ):
        """
        @brief initialization
        Be aware that all scalars must be python type instead of tensors.
        Otherwise, GPU version can be weirdly slow.
        @param node_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order
        @param node_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order
        @param movable_macro_mask some large movable macros need to be scaled to avoid halos
        @param bin_center_x bin center x locations
        @param bin_center_y bin center y locations
        @param target_density target density
        @param xl left boundary
        @param yl bottom boundary
        @param xh right boundary
        @param yh top boundary
        @param bin_size_x bin width
        @param bin_size_y bin height
        @param num_movable_nodes number of movable cells
        @param num_terminals number of fixed cells
        @param num_filler_nodes number of filler cells
        @param padding bin padding to boundary of placement region
        @param deterministic_flag control whether to use deterministic routine
        @param fast_mode if true, only gradient is computed, while objective computation is skipped
        """
        super(ElectricPotential,
              self).__init__(node_size_x=node_size_x,
                             node_size_y=node_size_y,
                             bin_center_x=bin_center_x,
                             bin_center_y=bin_center_y,
                             target_density=target_density,
                             xl=xl,
                             yl=yl,
                             xh=xh,
                             yh=yh,
                             bin_size_x=bin_size_x,
                             bin_size_y=bin_size_y,
                             num_movable_nodes=num_movable_nodes,
                             num_terminals=num_terminals,
                             num_filler_nodes=num_filler_nodes,
                             padding=padding,
                             deterministic_flag=deterministic_flag,
                             sorted_node_map=sorted_node_map,
                             movable_macro_mask=movable_macro_mask)
        self.fast_mode = fast_mode
        self.fence_regions = fence_regions
        self.fence_region_mask = fence_region_mask
        self.placedb = placedb
        self.target_density = target_density
        ## set by build_density_op func
        self.region_id = None
        self.filler_start_map = None
        self.filler_beg = None
        self.filler_end = None


    def compute_fence_region_map(self, fence_region, macro_pos_x=None, macro_pos_y=None, macro_size_x=None, macro_size_y=None):
        if(macro_pos_x is not None):
            pos = torch.cat([fence_region[:,0], macro_pos_x, fence_region[:,1], macro_pos_y], 0)
            num_terminals = fence_region.size(0) + macro_size_x.size(0)
            node_size_x = torch.cat([fence_region[:,2] - fence_region[:,0], macro_size_x], 0)
            node_size_y = torch.cat([fence_region[:,3] - fence_region[:,1], macro_size_y], 0)
        else:
            pos = fence_region[:,:2].t().contiguous().view(-1)
            num_terminals = fence_region.size(0)
            node_size_x = fence_region[:,2] - fence_region[:,0]
            node_size_y = fence_region[:,3] - fence_region[:,1]
        max_size_x = node_size_x.max()
        max_size_y = node_size_y.max()
        num_fixed_impacted_bins_x = ((max_size_x + self.bin_size_x) /
                                        self.bin_size_x).ceil().clamp(
                                            max=self.num_bins_x)
        num_fixed_impacted_bins_y = ((max_size_y + self.bin_size_y) /
                                        self.bin_size_y).ceil().clamp(
                                            max=self.num_bins_y)

        if pos.is_cuda:
            func = electric_potential_cuda.fixed_density_map
        else:
            func = electric_potential_cpp.fixed_density_map

        fence_region_map = func(
            pos, node_size_x, node_size_y, self.bin_center_x,
            self.bin_center_y, self.xl, self.yl, self.xh, self.yh,
            self.bin_size_x, self.bin_size_y, 0,
            num_terminals, self.num_bins_x, self.num_bins_y,
            num_fixed_impacted_bins_x, num_fixed_impacted_bins_y,
            self.deterministic_flag)
        # scale density of virtual macros
        ## target density for fence regions may differ from global target density
        ## recompute target density according fence region utilization
        # area for cells in this region
        total_movable_node_area = (self.node_size_x[:self.num_movable_nodes][~self.fence_region_mask]*self.node_size_y[:self.num_movable_nodes][~self.fence_region_mask]).sum().item()
        ## area for other cells
        other_movable_nodes_area = (self.node_size_x[:self.num_movable_nodes][self.fence_region_mask]*self.node_size_y[:self.num_movable_nodes][self.fence_region_mask]).sum().item()

        if(self.region_id < len(self.placedb.regions)):
            terminal_NI_area = (self.node_size_x[self.num_movable_nodes+self.num_terminals:self.num_movable_nodes+self.num_terminals+self.placedb.num_terminal_NIs]*self.node_size_y[self.num_movable_nodes+self.num_terminals:self.num_movable_nodes+self.num_terminals+self.placedb.num_terminal_NIs]).sum().item()
            print("terminal_NI_area:", terminal_NI_area)
            fence_region_area = (node_size_x*node_size_y).sum().item() - terminal_NI_area
        else:
            fence_region_area = (node_size_x*node_size_y).sum().item()
        total_space_area = (self.xh-self.xl)*(self.yh-self.yl) - fence_region_area

        utilization = min(total_movable_node_area / total_space_area, 1.0)
        self.target_density = max(utilization + 0.01, self.target_density)
        fence_region_map.mul_(self.target_density)
        self.fence_region_map = fence_region_map
        return fence_region_map

    def reset(self):
        """ Compute members derived from input
        """
        super(ElectricPotential, self).reset()
        logger.info("regard %d cells as movable macros in global placement" %
                    (self.num_movable_macros))

        self.exact_expkM = None
        self.exact_expkN = None
        self.inv_wu2_plus_wv2 = None
        self.wu_by_wu2_plus_wv2_half = None
        self.wv_by_wu2_plus_wv2_half = None

        # dct2, idct2, idct_idxst, idxst_idct functions
        self.dct2 = None
        self.idct2 = None
        self.idct_idxst = None
        self.idxst_idct = None

        # reset fence region map
        self.fence_region_map = None
        self.num_remove_filler = None

    def calc_num_remove_filler(self):
        '''
        @description: remove some fillers to balance density
        @param {type}
        @return:
        '''
        if(self.placedb.num_filler_nodes == 0):
            return 0
        area = (self.xh-self.xl)*(self.yh-self.yl)
        num_movable_nodes = self.num_movable_nodes
        total_movable_node_area = (self.node_size_x[:num_movable_nodes]*self.node_size_y[:num_movable_nodes]).masked_fill(self.fence_region_mask, 0).sum().data.item()
        fence_region_area = ((self.fence_regions[:,2]-self.fence_regions[:,0])*(self.fence_regions[:,3]-self.fence_regions[:,1])).sum().item()
        if(self.region_id < len(self.placedb.regions)):
            terminal_NI_area = (self.node_size_x[self.num_movable_nodes+self.num_terminals:self.num_movable_nodes+self.num_terminals+self.placedb.num_terminal_NIs]*self.node_size_y[self.num_movable_nodes+self.num_terminals:self.num_movable_nodes+self.num_terminals+self.placedb.num_terminal_NIs]).sum().item()
            print("terminal_NI_area:", terminal_NI_area)
            fence_region_area -= terminal_NI_area
        # total_terminal_area = (fence_region_density_mask | initial_density_mask).float().sum()*bin_size_x*bin_size_y
        # total_filler_area = max(0,area - total_movable_node_area - total_terminal_area)
        # num_nodes = pos.size(0)//2
        # print(area, self.placedb.area, self.placedb.total_fixed_node_area, fence_region_area, self.placedb.total_space_area)

        placeable_area = self.placedb.area - (self.placedb.total_fixed_node_area + fence_region_area)

        total_filler_node_area = max(placeable_area*self.target_density.item()-total_movable_node_area, 0.0)

        filler_size_x = self.node_size_x[-1].item()
        filler_size_y = self.node_size_y[-1].item()
        num_filler_reduced = min(max(0,int(round(total_filler_node_area/(filler_size_x*filler_size_y)))), self.num_filler_nodes)

        num_remove_filler = self.num_filler_nodes - num_filler_reduced
        self.num_remove_filler = num_remove_filler
        # print("remove filler:", num_remove_filler)
        return num_remove_filler


    def forward(self, pos):
        if self.initial_density_map is None:
            if(self.fence_regions is not None):
                self.fence_region_map = self.compute_fence_region_map(self.fence_regions)
                self.num_remove_filler = self.calc_num_remove_filler()
            else:
                self.fence_region_map = None
            num_nodes = self.placedb.num_nodes

            if(self.fence_regions is not None):
                if(self.num_terminals > 0):
                    self.initial_density_map = self.compute_fence_region_map(
                    self.fence_regions,
                    pos[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals],
                    pos[num_nodes+self.num_movable_nodes:num_nodes+self.num_movable_nodes+self.num_terminals],
                    self.node_size_x[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals],
                    self.node_size_y[self.num_movable_nodes:self.num_movable_nodes+self.num_terminals]
                    )
                else:
                    self.initial_density_map = self.compute_fence_region_map(self.fence_regions)
            else:
                self.compute_initial_density_map(pos)
            # self.compute_initial_density_map(pos)
            # plot(0, self.initial_density_map.clone().div(self.bin_size_x*self.bin_size_y).cpu().numpy(), self.padding, 'summary/initial_potential_map')
            logger.info("fixed density map: average %g, max %g, bin area %g" %
                        (self.initial_density_map.mean(),
                         self.initial_density_map.max(),
                         self.bin_size_x * self.bin_size_y))

            # expk
            M = self.num_bins_x
            N = self.num_bins_y
            self.exact_expkM = precompute_expk(M,
                                               dtype=pos.dtype,
                                               device=pos.device)
            self.exact_expkN = precompute_expk(N,
                                               dtype=pos.dtype,
                                               device=pos.device)

            # init dct2, idct2, idct_idxst, idxst_idct with expkM and expkN
            self.dct2 = dct.DCT2(self.exact_expkM, self.exact_expkN)
            if not self.fast_mode:
                self.idct2 = dct.IDCT2(self.exact_expkM, self.exact_expkN)
            self.idct_idxst = dct.IDCT_IDXST(self.exact_expkM,
                                             self.exact_expkN)
            self.idxst_idct = dct.IDXST_IDCT(self.exact_expkM,
                                             self.exact_expkN)

            # wu and wv
            wu = torch.arange(M, dtype=pos.dtype, device=pos.device).mul(
                2 * np.pi / M).view([M, 1])
            # scale wv because the aspect ratio of a bin may not be 1
            wv = torch.arange(N, dtype=pos.dtype,
                              device=pos.device).mul(2 * np.pi / N).view(
                                  [1,
                                   N]).mul_(self.bin_size_x / self.bin_size_y)
            wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
            wu2_plus_wv2[0,
                         0] = 1.0  # avoid zero-division, it will be zeroed out
            self.inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
            self.inv_wu2_plus_wv2[0, 0] = 0.0
            self.wu_by_wu2_plus_wv2_half = wu.mul(self.inv_wu2_plus_wv2).mul_(
                1. / 2)
            self.wv_by_wu2_plus_wv2_half = wv.mul(self.inv_wu2_plus_wv2).mul_(
                1. / 2)

        return ElectricPotentialFunction.apply(
            pos, self.node_size_x_clamped, self.node_size_y_clamped,
            self.offset_x, self.offset_y, self.ratio, self.bin_center_x,
            self.bin_center_y, self.initial_density_map, self.target_density,
            self.xl, self.yl, self.xh, self.yh, self.bin_size_x,
            self.bin_size_y, self.num_movable_nodes, self.num_filler_nodes,
            self.padding, self.padding_mask, self.num_bins_x, self.num_bins_y,
            self.num_movable_impacted_bins_x, self.num_movable_impacted_bins_y,
            self.num_filler_impacted_bins_x, self.num_filler_impacted_bins_y,
            self.deterministic_flag, self.sorted_node_map, self.exact_expkM,
            self.exact_expkN, self.inv_wu2_plus_wv2,
            self.wu_by_wu2_plus_wv2_half, self.wv_by_wu2_plus_wv2_half,
            self.dct2, self.idct2, self.idct_idxst, self.idxst_idct,
            self.fast_mode, self.fence_region_map, self.fence_region_mask, self.filler_beg, self.filler_end)
