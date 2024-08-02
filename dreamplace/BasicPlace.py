# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##
# @file   BasicPlace.py
# @author Yibo Lin
# @date   Jun 2018
# @brief  Base placement class
#

from dataclasses import dataclass, fields
import os
import sys
import time
import gzip

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import re
import numpy as np
import logging
import torch
import torch.nn as nn
import dreamplace.ops.move_boundary.move_boundary as move_boundary
import dreamplace.ops.hpwl.hpwl as hpwl
import dreamplace.ops.rmst_wl.rmst_wl as rmst_wl
import dreamplace.ops.macro_legalize.macro_legalize as macro_legalize
import dreamplace.ops.greedy_legalize.greedy_legalize as greedy_legalize
import dreamplace.ops.abacus_legalize.abacus_legalize as abacus_legalize
import dreamplace.ops.legality_check.legality_check as legality_check
import dreamplace.ops.draw_place.draw_place as draw_place
import dreamplace.ops.pin_pos.pin_pos as pin_pos
import dreamplace.ops.global_swap.global_swap as global_swap
import dreamplace.ops.k_reorder.k_reorder as k_reorder
import dreamplace.ops.independent_set_matching.independent_set_matching as independent_set_matching
import pdb


@dataclass
class FloorplanInfo:
    xl: float
    yl: float
    xh: float
    yh: float
    site_width: float
    row_height: float
    scale_factor: float
    routing_grid_xl: float
    routing_grid_yl: float
    routing_grid_xh: float
    routing_grid_yh: float
    routing_V: float
    routing_H: float
    macro_util_V: torch.Tensor
    macro_util_H: torch.Tensor
    macro_padding_x: float
    macro_padding_y: float
    bndry_padding_x: float
    bndry_padding_y: float

    def scale(self, factor):
        for field in fields(self):
            value = getattr(self, field.name)
            setattr(self, field.name, value * factor)


class PlaceDataCollection(object):
    """
    @brief A wraper for all data tensors on device for building ops
    """

    def __init__(self, pos, orient_logits, params, placedb, device):
        """
        @brief initialization
        @param pos locations of cells
        @param params parameters
        @param placedb placement database
        @param device cpu or cuda
        """
        self.device = device
        # position should be parameter
        self.pos = pos
        # orient logits should also be parameter
        self.orient_logits = orient_logits

        with torch.no_grad():
            # other tensors required to build ops
            self.best_theta = torch.zeros(self.orient_logits.size(0), dtype=self.orient_logits.dtype, device=device)
            self.node_size_x = torch.from_numpy(placedb.node_size_x).to(device)
            self.node_size_y = torch.from_numpy(placedb.node_size_y).to(device)
            # original node size for legalization, since they will be adjusted in global placement
            if params.routability_opt_flag:
                self.original_node_size_x = self.node_size_x.clone()
                self.original_node_size_y = self.node_size_y.clone()

            self.pin_offset_x = torch.tensor(
                placedb.pin_offset_x, dtype=self.pos[0].dtype, device=device
            )
            self.pin_offset_y = torch.tensor(
                placedb.pin_offset_y, dtype=self.pos[0].dtype, device=device
            )
            # original pin offset for legalization, since they will be adjusted in global placement
            if params.routability_opt_flag:
                self.original_pin_offset_x = self.pin_offset_x.clone()
                self.original_pin_offset_y = self.pin_offset_y.clone()

            self.target_density = torch.empty(1, dtype=self.pos[0].dtype, device=device)
            self.target_density.data.fill_(params.target_density)

            self.node_areas = self.node_size_x * self.node_size_y
            self.movable_macro_mask = torch.from_numpy(placedb.movable_macro_mask).to(
                device
            )
            self.fixed_macro_mask = torch.from_numpy(placedb.fixed_macro_mask).to(
                device
            )

            self.pin2node_map = torch.from_numpy(placedb.pin2node_map).to(device)
            self.flat_node2pin_map = torch.from_numpy(placedb.flat_node2pin_map).to(
                device
            )
            self.flat_node2pin_start_map = torch.from_numpy(
                placedb.flat_node2pin_start_map
            ).to(device)
            # number of pins for each cell
            self.pin_weights = (
                self.flat_node2pin_start_map[1:] - self.flat_node2pin_start_map[:-1]
            ).to(self.node_size_x.dtype)

            self.unit_pin_capacity = torch.empty(
                1, dtype=self.pos[0].dtype, device=device
            )
            self.unit_pin_capacity.data.fill_(params.unit_pin_capacity)
            if params.routability_opt_flag:
                unit_pin_capacity = (
                    self.pin_weights[: placedb.num_movable_nodes]
                    / self.node_areas[: placedb.num_movable_nodes]
                )
                avg_pin_capacity = unit_pin_capacity.mean() * self.target_density
                # min(computed, params.unit_pin_capacity)
                self.unit_pin_capacity = avg_pin_capacity.clamp_(
                    max=params.unit_pin_capacity
                )
                logging.info("unit_pin_capacity = %g" % (self.unit_pin_capacity))

            # routing information
            # project initial routing utilization map to one layer
            self.initial_horizontal_utilization_map = None
            self.initial_vertical_utilization_map = None
            if (
                params.routability_opt_flag
                and placedb.initial_horizontal_demand_map is not None
            ):
                self.initial_horizontal_utilization_map = (
                    torch.from_numpy(placedb.initial_horizontal_demand_map)
                    .to(device)
                    .div_(
                        placedb.routing_grid_size_y * placedb.unit_horizontal_capacity
                    )
                )
                self.initial_vertical_utilization_map = (
                    torch.from_numpy(placedb.initial_vertical_demand_map)
                    .to(device)
                    .div_(placedb.routing_grid_size_x * placedb.unit_vertical_capacity)
                )

            self.pin2net_map = torch.from_numpy(placedb.pin2net_map).to(device)
            self.flat_net2pin_map = torch.from_numpy(placedb.flat_net2pin_map).to(
                device
            )
            self.flat_net2pin_start_map = torch.from_numpy(
                placedb.flat_net2pin_start_map
            ).to(device)
            if np.amin(placedb.net_weights) != np.amax(
                placedb.net_weights
            ):  # weights are meaningful
                self.net_weights = torch.from_numpy(placedb.net_weights).to(device)
            else:  # an empty tensor
                logging.warning("net weights are all the same, ignored")
                self.net_weights = torch.Tensor().to(device)

            # regions
            self.flat_region_boxes = torch.from_numpy(placedb.flat_region_boxes).to(
                device
            )
            self.flat_region_boxes_start = torch.from_numpy(
                placedb.flat_region_boxes_start
            ).to(device)
            self.node2fence_region_map = torch.from_numpy(
                placedb.node2fence_region_map
            ).to(device)
            if len(placedb.regions) > 0:
                # This is for multi-electric potential and legalization
                # boxes defined as left-bottm point and top-right point
                self.virtual_macro_fence_region = [
                    torch.from_numpy(region).to(device)
                    for region in placedb.virtual_macro_fence_region
                ]
                ## this is for overflow op
                self.total_movable_node_area_fence_region = torch.from_numpy(
                    placedb.total_movable_node_area_fence_region
                ).to(device)
                ## this is for gamma update
                self.num_movable_nodes_fence_region = torch.from_numpy(
                    placedb.num_movable_nodes_fence_region
                ).to(device)
                ## this is not used yet
                self.num_filler_nodes_fence_region = torch.from_numpy(
                    placedb.num_filler_nodes_fence_region
                ).to(device)

            self.net_mask_all = torch.from_numpy(
                np.ones(placedb.num_nets, dtype=np.uint8)
            ).to(
                device
            )  # all nets included
            net_degrees = np.ediff1d(placedb.flat_net2pin_start_map)
            net_mask = np.logical_and(
                2 <= net_degrees, net_degrees < params.ignore_net_degree
            ).astype(np.uint8)
            self.net_mask_ignore_large_degrees = torch.from_numpy(net_mask).to(
                device
            )  # nets with large degrees are ignored
            large_weight_net_mask = placedb.net_weights < params.ignore_net_weight
            self.net_mask_ignore_large_weights = torch.from_numpy(
                large_weight_net_mask.astype(np.uint8)
            ).to(device)
            # number of pins for each node
            self.num_pins_in_nodes = torch.zeros_like(self.node_size_x)
            self.num_pins_in_nodes[: placedb.num_physical_nodes] = self.pin_weights

            # avoid computing gradient for fixed macros
            # 1 is for fixed macros
            self.pin_mask_ignore_fixed_macros = (
                self.pin2node_map >= placedb.num_movable_nodes
            )

            # sort nodes by size, return their sorted indices, designed for memory coalesce in electrical force
            movable_size_x = self.node_size_x[: placedb.num_movable_nodes]
            _, self.sorted_node_map = torch.sort(movable_size_x)
            self.sorted_node_map = self.sorted_node_map.to(torch.int32)
            # self.sorted_node_map = torch.arange(0, placedb.num_movable_nodes, dtype=torch.int32, device=device)

            # store floorplan info for later rescaling during legalization/detailed placement
            self.fp_info = FloorplanInfo(
                placedb.xl,
                placedb.yl,
                placedb.xh,
                placedb.yh,
                placedb.site_width,
                placedb.row_height,
                params.scale_factor,
                placedb.routing_grid_xl,
                placedb.routing_grid_yl,
                placedb.routing_grid_xh,
                placedb.routing_grid_yh,
                placedb.routing_V,
                placedb.routing_H,
                torch.from_numpy(placedb.macro_util_V).to(device),
                torch.from_numpy(placedb.macro_util_H).to(device),
                placedb.macro_padding_x,
                placedb.macro_padding_y,
                placedb.bndry_padding_x,
                placedb.bndry_padding_y,
            )

    def bin_center_x_padded(self, placedb, padding, num_bins_x):
        """
        @brief compute array of bin center horizontal coordinates with padding
        @param placedb placement database
        @param padding number of bins padding to boundary of placement region
        """
        bin_size_x = (self.fp_info.xh - self.fp_info.xl) / num_bins_x
        xl = self.fp_info.xl - padding * bin_size_x
        xh = self.fp_info.xh + padding * bin_size_x
        bin_center_x = torch.from_numpy(placedb.bin_centers(xl, xh, bin_size_x)).to(
            self.device
        )
        return bin_center_x

    def bin_center_y_padded(self, placedb, padding, num_bins_y):
        """
        @brief compute array of bin center vertical coordinates with padding
        @param placedb placement database
        @param padding number of bins padding to boundary of placement region
        """
        bin_size_y = (self.fp_info.yh - self.fp_info.yl) / num_bins_y
        yl = self.fp_info.yl - padding * bin_size_y
        yh = self.fp_info.yh + padding * bin_size_y
        bin_center_y = torch.from_numpy(placedb.bin_centers(yl, yh, bin_size_y)).to(
            self.device
        )
        return bin_center_y

    # def scale(self, factor):
    #     # TODO: add regions
    #     with torch.no_grad():
    #         self.pos.mul_(factor).round_()
    #         self.node_size_x.mul_(factor).round_()
    #         self.node_size_y.mul_(factor).round_()
    #         self.original_node_size_x.mul_(factor).round_()
    #         self.original_node_size_y.mul_(factor).round_()
    #         self.pin_offset_x.mul_(factor)
    #         self.pin_offset_y.mul_(factor)
    #         self.original_pin_offset_x.mul_(factor)
    #         self.original_pin_offset_y.mul_(factor)
    #         self.unit_pin_capacity.div_(factor * factor)
    #         self.flat_region_boxes.mul_(factor).round_()
    #         self.node_areas.mul_(factor * factor)
    #         self.fp_info.scale(factor)
    #         self.fp_info.scale_factor = factor


class PlaceOpCollection(object):
    """
    @brief A wrapper for all ops
    """

    def __init__(self):
        """
        @brief initialization
        """
        self.pin_pos_op = None
        self.move_boundary_op = None
        self.hpwl_op = None
        self.weight_hpwl_op = None
        self.rsmt_wl_op = None
        self.density_overflow_op = None
        self.legality_check_op = None
        self.legalize_op = None
        self.detailed_place_op = None
        self.wirelength_op = None
        self.update_gamma_op = None
        self.density_op = None
        self.update_density_weight_op = None
        self.precondition_op = None
        self.noise_op = None
        self.draw_place_op = None
        self.route_utilization_map_op = None
        self.pin_utilization_map_op = None
        self.nctugr_congestion_map_op = None
        self.adjust_node_area_op = None
        self.macro_overlap_op = None
        self.update_macro_overlap_weight_op = None
        self.macro_refinement_op = None
        self.net_crossing_op = None


class BasicPlace(nn.Module):
    """
    @brief Base placement class.
    All placement engines should be derived from this class.
    """

    def __init__(self, params, placedb):
        """
        @brief initialization
        @param params parameter
        @param placedb placement database
        """
        super(BasicPlace, self).__init__()

        tt = time.time()
        self.init_pos = np.zeros(placedb.num_nodes * 2, dtype=placedb.dtype)

        # initial location of cells
        init_loc_perc_x = params.init_loc_perc_x
        init_loc_perc_y = params.init_loc_perc_y
        if (
            not 0.0 < params.init_loc_perc_x < 1.0
            or not 0.0 < params.init_loc_perc_y < 1.0
        ):
            init_loc_perc_x = 0.5
            init_loc_perc_y = 0.5
            logging.warn("incorrect initial location provided, choose center of layout")

        init_loc_x = (
            placedb.xl * 1.0 + (placedb.xh * 1.0 - placedb.xl * 1.0) * init_loc_perc_x
        )
        init_loc_y = (
            placedb.yl * 1.0 + (placedb.yh * 1.0 - placedb.yl * 1.0) * init_loc_perc_y
        )

        # x position
        self.init_pos[0 : placedb.num_physical_nodes] = placedb.node_x
        if params.global_place_flag and params.random_center_init_flag:
            logging.info(
                f"move cells to location {init_loc_x, init_loc_y} with random noise"
            )
            self.init_pos[0 : placedb.num_movable_nodes] = (
                np.random.normal(
                    loc=init_loc_x,
                    scale=(placedb.xh - placedb.xl) * 0.001,
                    size=placedb.num_movable_nodes,
                )
                - placedb.node_size_x[0 : placedb.num_movable_nodes] / 2
            )

        # y position
        self.init_pos[
            placedb.num_nodes : placedb.num_nodes + placedb.num_physical_nodes
        ] = placedb.node_y
        if params.global_place_flag and params.random_center_init_flag:
            self.init_pos[
                placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes
            ] = (
                np.random.normal(
                    loc=init_loc_y,
                    scale=(placedb.yh - placedb.yl) * 0.001,
                    size=placedb.num_movable_nodes,
                )
                - placedb.node_size_y[0 : placedb.num_movable_nodes] / 2
            )

        if placedb.num_filler_nodes:  # uniformly distribute filler cells in the layout
            if len(placedb.regions) > 0:
                ### uniformly spread fillers in fence region
                ### for cells in the fence region
                for i, region in enumerate(placedb.regions):
                    filler_beg, filler_end = placedb.filler_start_map[i : i + 2]
                    subregion_areas = (region[:, 2] - region[:, 0]) * (
                        region[:, 3] - region[:, 1]
                    )
                    total_area = np.sum(subregion_areas)
                    subregion_area_ratio = subregion_areas / total_area
                    subregion_num_filler = np.round(
                        (filler_end - filler_beg) * subregion_area_ratio
                    )
                    subregion_num_filler[-1] = (filler_end - filler_beg) - np.sum(
                        subregion_num_filler[:-1]
                    )
                    subregion_num_filler_start_map = np.concatenate(
                        [np.zeros([1]), np.cumsum(subregion_num_filler)], 0
                    ).astype(np.int32)
                    for j, subregion in enumerate(region):
                        sub_filler_beg, sub_filler_end = subregion_num_filler_start_map[
                            j : j + 2
                        ]
                        self.init_pos[
                            placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_beg : placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_end
                        ] = np.random.uniform(
                            low=subregion[0],
                            high=subregion[2] - placedb.filler_size_x_fence_region[i],
                            size=sub_filler_end - sub_filler_beg,
                        )
                        self.init_pos[
                            placedb.num_nodes
                            + placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_beg : placedb.num_nodes
                            + placedb.num_physical_nodes
                            + filler_beg
                            + sub_filler_end
                        ] = np.random.uniform(
                            low=subregion[1],
                            high=subregion[3] - placedb.filler_size_y_fence_region[i],
                            size=sub_filler_end - sub_filler_beg,
                        )

                ### for cells outside fence region
                filler_beg, filler_end = placedb.filler_start_map[-2:]
                self.init_pos[
                    placedb.num_physical_nodes
                    + filler_beg : placedb.num_physical_nodes
                    + filler_end
                ] = np.random.uniform(
                    low=placedb.xl,
                    high=placedb.xh - placedb.filler_size_x_fence_region[-1],
                    size=filler_end - filler_beg,
                )
                self.init_pos[
                    placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_beg : placedb.num_nodes
                    + placedb.num_physical_nodes
                    + filler_end
                ] = np.random.uniform(
                    low=placedb.yl,
                    high=placedb.yh - placedb.filler_size_y_fence_region[-1],
                    size=filler_end - filler_beg,
                )

            else:
                self.init_pos[
                    placedb.num_physical_nodes : placedb.num_nodes
                ] = np.random.uniform(
                    low=placedb.xl,
                    high=placedb.xh - placedb.node_size_x[-placedb.num_filler_nodes],
                    size=placedb.num_filler_nodes,
                )
                self.init_pos[
                    placedb.num_nodes
                    + placedb.num_physical_nodes : placedb.num_nodes * 2
                ] = np.random.uniform(
                    low=placedb.yl,
                    high=placedb.yh - placedb.node_size_y[-placedb.num_filler_nodes],
                    size=placedb.num_filler_nodes,
                )

        logging.debug("prepare init_pos takes %.2f seconds" % (time.time() - tt))

        # setting device
        if params.gpu and torch.cuda.is_available():
            if params.gpu_id >= torch.cuda.device_count():
                params.gpu_id = 0
            torch.cuda.set_device(params.gpu_id)
            self.device = torch.device("cuda")
            logging.info(
                f"Using Torch GPU device # {params.gpu_id}: {torch.cuda.get_device_name(params.gpu_id)}"
            )
        else:
            self.device = torch.device("cpu")
            logging.info("Using Torch CPU device")

        # position should be parameter
        # must be defined in BasicPlace
        tt = time.time()
        self.pos = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(self.init_pos).to(self.device))]
        )
        logging.debug("build pos takes %.2f seconds" % (time.time() - tt))

        # Orientation logits
        tt = time.time()
        self.orient_logits = nn.Parameter(torch.from_numpy(placedb.orient_logits).to(self.device))
        logging.debug("build orient_logits takes %.2f seconds" % (time.time() - tt))

        # shared data on device for building ops
        # I do not want to construct the data from placedb again and again for each op
        tt = time.time()
        self.data_collections = PlaceDataCollection(
            self.pos, self.orient_logits, params, placedb, self.device
        )
        logging.debug("build data_collections takes %.2f seconds" % (time.time() - tt))

        # similarly I wrap all ops
        tt = time.time()
        self.op_collections = PlaceOpCollection()
        logging.debug("build op_collections takes %.2f seconds" % (time.time() - tt))

        tt = time.time()
        # position to pin position
        self.op_collections.pin_pos_op = self.build_pin_pos(
            params, placedb, self.data_collections, self.device
        )
        # bound nodes to layout region
        self.op_collections.move_boundary_op = self.build_move_boundary(
            params, placedb, self.data_collections, self.device
        )
        # hpwl and density overflow ops for evaluation
        self.op_collections.hpwl_op = self.build_hpwl(
            params,
            placedb,
            self.data_collections,
            self.op_collections.pin_pos_op,
            self.device,
        )
        # hpwl for nets with smaller weight than ignore_net_weight
        self.op_collections.weight_hpwl_op = self.build_weight_hpwl(
            params,
            placedb,
            self.data_collections,
            self.op_collections.pin_pos_op,
            self.device,
        )
        # rectilinear minimum steiner tree wirelength from flute
        # can only be called once
        self.op_collections.rsmt_wl_op = self.build_rsmt_wl(
            params,
            placedb,
            self.data_collections,
            self.op_collections.pin_pos_op,
            torch.device("cpu"),
        )
        # legality check
        self.op_collections.legality_check_op = self.build_legality_check(
            params, placedb, self.data_collections, self.device
        )
        # legalization
        if len(placedb.regions) > 0:
            (
                self.op_collections.legalize_op,
                self.op_collections.individual_legalize_op,
            ) = self.build_multi_fence_region_legalization(
                params, placedb, self.data_collections, self.device
            )
        else:
            self.op_collections.legalize_op = self.build_legalization(
                params, placedb, self.data_collections, self.device
            )
        # detailed placement
        self.op_collections.detailed_place_op = self.build_detailed_placement(
            params, placedb, self.data_collections, self.device
        )
        # draw placement
        self.op_collections.draw_place_op = self.build_draw_placement(
            params, placedb, self.data_collections
        )

        # flag for rsmt_wl_op
        # can only read once
        self.read_lut_flag = True

        logging.debug("build BasicPlace ops takes %.2f seconds" % (time.time() - tt))

    def __call__(self, params, placedb):
        """
        @brief Solve placement.
        placeholder for derived classes.
        @param params parameters
        @param placedb placement database
        """
        pass

    
    def update_best_theta(self):
        choices = torch.argmax(nn.functional.gumbel_softmax(self.orient_logits, tau=1, hard=True), dim=1)
        # self.data_collections.best_theta = choices * np.pi / 2
        # update in-place
        self.data_collections.best_theta.copy_(choices * np.pi / 2)
    
    
    def build_pin_pos(self, params, placedb, data_collections, device):
        """
        @brief sum up the pins for each cell
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return pin_pos.PinPos(
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            pin2node_map=data_collections.pin2node_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            num_physical_nodes=placedb.num_physical_nodes,
            h=data_collections.node_size_y,
            w=data_collections.node_size_x,
            algorithm="node-by-node",
            orient_logits=data_collections.orient_logits,
            best_theta=data_collections.best_theta,
        )

    def build_move_boundary(self, params, placedb, data_collections, device):
        """
        @brief bound nodes into layout region
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return move_boundary.MoveBoundary(
            data_collections.node_size_x,
            data_collections.node_size_y,
            fp_info=data_collections.fp_info,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
        )

    def build_hpwl(self, params, placedb, data_collections, pin_pos_op, device):
        """
        @brief compute half-perimeter wirelength
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        @param device cpu or cuda
        """

        wirelength_for_pin_op = hpwl.HPWL(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_degrees,  # net_mask_all
            algorithm="net-by-net", # "atomic"
        )

        # wirelength for position
        def build_wirelength_op(pos, reduction=True):
            hpwls = wirelength_for_pin_op(pin_pos_op(pos))
            if reduction:
                return hpwls.sum() / data_collections.fp_info.scale_factor
            else:
                return hpwls / data_collections.fp_info.scale_factor

        return build_wirelength_op

    def build_weight_hpwl(self, params, placedb, data_collections, pin_pos_op, device):
        """
        @brief compute half-perimeter wirelength for weights less than ignore_net_weight
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param pin_pos_op the op to compute pin locations according to cell locations
        @param device cpu or cuda
        """

        wirelength_for_pin_op = hpwl.HPWL(
            flat_netpin=data_collections.flat_net2pin_map,
            netpin_start=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            net_weights=data_collections.net_weights,
            net_mask=data_collections.net_mask_ignore_large_weights,
            algorithm="net-by-net", # "atomic"
        )

        # wirelength for position
        def build_wirelength_op(pos, reduction=True):
            hpwls = wirelength_for_pin_op(pin_pos_op(pos))
            if reduction:
                return hpwls.sum() / data_collections.fp_info.scale_factor
            else:
                return hpwls / data_collections.fp_info.scale_factor

        return build_wirelength_op

    def build_rsmt_wl(self, params, placedb, data_collections, pin_pos_op, device):
        """
        @brief compute rectilinear steiner minimal tree wirelength with flute
        @param params parameters
        @param placedb placement database
        @param pin_pos_op the op to compute pin locations according to cell locations
        @param device cpu or cuda
        """
        # wirelength cost
        POWVFILE = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../thirdparty/NCTUgr.ICCAD2012/POWV9.dat"
            )
        )
        POSTFILE = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "../thirdparty/NCTUgr.ICCAD2012/POST9.dat"
            )
        )
        logging.info("POWVFILE = %s" % (POWVFILE))
        logging.info("POSTFILE = %s" % (POSTFILE))
        wirelength_for_pin_op = rmst_wl.RmstWL(
            flat_netpin=torch.from_numpy(placedb.flat_net2pin_map).to(device),
            netpin_start=torch.from_numpy(placedb.flat_net2pin_start_map).to(device),
            ignore_net_degree=params.ignore_net_degree,
            POWVFILE=POWVFILE,
            POSTFILE=POSTFILE,
        )

        # wirelength for position
        def build_wirelength_op(pos, reduction=True):
            pin_pos = pin_pos_op(pos)
            wls = wirelength_for_pin_op(pin_pos.clone().cpu(), self.read_lut_flag)
            self.read_lut_flag = False
            if reduction:
                return wls.sum() / data_collections.fp_info.scale_factor
            else:
                return wls / data_collections.fp_info.scale_factor

        return build_wirelength_op

    def build_legality_check(self, params, placedb, data_collections, device):
        """
        @brief legality check
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        return legality_check.LegalityCheck(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_terminals=placedb.num_terminals,
            num_movable_nodes=placedb.num_movable_nodes,
        )

    def build_legalization(self, params, placedb, data_collections, device):
        """
        @brief legalization
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        # for movable macro legalization
        # the number of bins control the search granularity
        top_ml = macro_legalize.MacroLegalize(
            node_size_x=data_collections.node_size_x[placedb.top_nodes_idx], # per layer
            node_size_y=data_collections.node_size_y[placedb.top_nodes_idx], # per layer
            node_weights=data_collections.num_pins_in_nodes[placedb.top_nodes_idx], # per layer
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_top_movable_nodes, 
            num_terminal_NIs=0,
            num_filler_nodes=0
        )
        btm_ml = macro_legalize.MacroLegalize(
            node_size_x=data_collections.node_size_x[placedb.btm_nodes_idx], # per layer
            node_size_y=data_collections.node_size_y[placedb.btm_nodes_idx], # per layer
            node_weights=data_collections.num_pins_in_nodes[placedb.btm_nodes_idx], # per layer
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_btm_movable_nodes,
            num_terminal_NIs=0,
            num_filler_nodes=0,
        )
        ml = macro_legalize.MacroLegalize(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
        )
        # for standard cell legalization
        # legalize_alg = mg_legalize.MGLegalize
        legalize_alg = greedy_legalize.GreedyLegalize
        gl = legalize_alg(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_bins_x=1,
            num_bins_y=64,
            # num_bins_x=64, num_bins_y=64,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
        )
        # for standard cell legalization
        al = abacus_legalize.AbacusLegalize(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_bins_x=1,
            num_bins_y=64,
            # num_bins_x=64, num_bins_y=64,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
        )

        def build_legalization_op_single_layer(pos):
            logging.info("Start legalization")
            pos1 = ml(pos, pos)
            pos2 = gl(pos1, pos1)
            legal = self.op_collections.legality_check_op(pos2)
            if not legal:
                logging.error("legality check failed in greedy legalization")
                return pos2
            return al(pos1, pos2)
        
        def build_legalization_op(pos):
            logging.info("Start legalization")
            top_pos = pos.view(2, -1)[:, placedb.top_nodes_idx].view(-1)
            pos1 = top_ml(top_pos, top_pos) # pass in the view of pos just having that side
            pos2 = btm_ml(pos1, pos1)
            # put pos2 back to pos
            pos_clone = pos.clone()
            pos_clone.view(2, -1)[:, placedb.top_nodes_idx] = pos2.view(2, -1)
            pos = pos_clone

            # pos3 = gl(pos2, pos2)
            # legal = self.op_collections.legality_check_op(pos3)
            # if not legal:
            #     logging.error("legality check failed in greedy legalization")
            #     return pos3
            # abacus doesn't work?
            # return al(pos2, pos3)
            return pos

        return build_legalization_op_single_layer

    def build_multi_fence_region_legalization(
        self, params, placedb, data_collections, device
    ):
        legal_ops = [
            self.build_fence_region_legalization(
                region_id, params, placedb, data_collections, device
            )
            for region_id in range(len(placedb.regions) + 1)
        ]

        pos_ml_list = []
        pos_gl_list = []

        def build_legalization_op(pos):
            for i in range(len(placedb.regions) + 1):
                pos, pos_ml, pos_gl = legal_ops[i][0](pos)
                pos_ml_list.append(pos_ml)
                pos_gl_list.append(pos_gl)
            legal = self.op_collections.legality_check_op(pos)
            if not legal:
                logging.error("legality check failed in greedy legalization")
                return pos
            else:
                ### start abacus legalizer
                for i in range(len(placedb.regions) + 1):
                    pos = legal_ops[i][1](pos, pos_ml_list[i], pos_gl_list[i])
            return pos

        def build_individual_legalization_ops(pos, region_id):
            pos = legal_ops[region_id][0](pos)[0]
            return pos

        return build_legalization_op, build_individual_legalization_ops

    def build_fence_region_legalization(
        self, region_id, params, placedb, data_collections, device
    ):
        ### reconstruct node size
        ### extract necessary nodes in the electric field and insert virtual macros to replace fence region
        num_nodes = placedb.num_nodes
        num_movable_nodes = placedb.num_movable_nodes
        num_filler_nodes = placedb.num_filler_nodes
        num_terminals = placedb.num_terminals
        num_terminal_NIs = placedb.num_terminal_NIs
        if region_id < len(placedb.regions):
            fence_region_mask = (
                data_collections.node2fence_region_map[:num_movable_nodes] == region_id
            )
        else:
            fence_region_mask = data_collections.node2fence_region_map[
                :num_movable_nodes
            ] >= len(placedb.regions)

        virtual_macros = data_collections.virtual_macro_fence_region[region_id]
        virtual_macros_center_x = (virtual_macros[:, 2] + virtual_macros[:, 0]) / 2
        virtual_macros_center_y = (virtual_macros[:, 3] + virtual_macros[:, 1]) / 2
        virtual_macros_size_x = (virtual_macros[:, 2] - virtual_macros[:, 0]).clamp(
            min=30
        )

        virtual_macros_size_y = (virtual_macros[:, 3] - virtual_macros[:, 1]).clamp(
            min=30
        )
        virtual_macros[:, 0] = virtual_macros_center_x - virtual_macros_size_x / 2
        virtual_macros[:, 1] = virtual_macros_center_y - virtual_macros_size_y / 2
        virtual_macros_pos = virtual_macros[:, 0:2].t().contiguous()

        ### node size
        node_size_x, node_size_y = (
            data_collections.node_size_x,
            data_collections.node_size_y,
        )
        filler_beg, filler_end = placedb.filler_start_map[region_id : region_id + 2]
        node_size_x = torch.cat(
            [
                node_size_x[:num_movable_nodes][fence_region_mask],  ## movable
                node_size_x[
                    num_movable_nodes : num_movable_nodes + num_terminals
                ],  ## terminals
                virtual_macros_size_x,  ## virtual macros
                node_size_x[
                    num_movable_nodes
                    + num_terminals : num_movable_nodes
                    + num_terminals
                    + num_terminal_NIs
                ],  ## terminal NIs
                node_size_x[
                    num_nodes
                    - num_filler_nodes
                    + filler_beg : num_nodes
                    - num_filler_nodes
                    + filler_end
                ],  ## fillers
            ],
            0,
        )
        node_size_y = torch.cat(
            [
                node_size_y[:num_movable_nodes][fence_region_mask],  ## movable
                node_size_y[
                    num_movable_nodes : num_movable_nodes + num_terminals
                ],  ## terminals
                virtual_macros_size_y,  ## virtual macros
                node_size_y[
                    num_movable_nodes
                    + num_terminals : num_movable_nodes
                    + num_terminals
                    + num_terminal_NIs
                ],  ## terminal NIs
                node_size_y[
                    num_nodes
                    - num_filler_nodes
                    + filler_beg : num_nodes
                    - num_filler_nodes
                    + filler_end
                ],  ## fillers
            ],
            0,
        )

        ### num pins in nodes
        ### 0 for virtual macros and fillers
        num_pins_in_nodes = data_collections.num_pins_in_nodes
        num_pins_in_nodes = torch.cat(
            [
                num_pins_in_nodes[:num_movable_nodes][fence_region_mask],  ## movable
                num_pins_in_nodes[
                    num_movable_nodes : num_movable_nodes + num_terminals
                ],  ## terminals
                torch.zeros(
                    virtual_macros_size_x.size(0),
                    dtype=num_pins_in_nodes.dtype,
                    device=device,
                ),  ## virtual macros
                num_pins_in_nodes[
                    num_movable_nodes
                    + num_terminals : num_movable_nodes
                    + num_terminals
                    + num_terminal_NIs
                ],  ## terminal NIs
                num_pins_in_nodes[
                    num_nodes
                    - num_filler_nodes
                    + filler_beg : num_nodes
                    - num_filler_nodes
                    + filler_end
                ],  ## fillers
            ],
            0,
        )
        ## num movable nodes and num filler nodes
        num_movable_nodes_fence_region = fence_region_mask.long().sum().item()
        num_filler_nodes_fence_region = filler_end - filler_beg
        num_terminals_fence_region = num_terminals + virtual_macros_size_x.size(0)
        assert (
            node_size_x.size(0)
            == node_size_y.size(0)
            == num_movable_nodes_fence_region
            + num_terminals_fence_region
            + num_terminal_NIs
            + num_filler_nodes_fence_region
        )

        ### flat region boxes
        flat_region_boxes = torch.tensor(
            [],
            device=node_size_x.device,
            dtype=data_collections.flat_region_boxes.dtype,
        )
        ### flat region boxes start
        flat_region_boxes_start = torch.tensor(
            [0],
            device=node_size_x.device,
            dtype=data_collections.flat_region_boxes_start.dtype,
        )
        ### node2fence region map: movable + terminal
        node2fence_region_map = torch.zeros(
            num_movable_nodes_fence_region + num_terminals_fence_region,
            dtype=data_collections.node2fence_region_map.dtype,
            device=node_size_x.device,
        ).fill_(data_collections.node2fence_region_map.max().item())

        ml = macro_legalize.MacroLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=num_pins_in_nodes,
            flat_region_boxes=flat_region_boxes,
            flat_region_boxes_start=flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_bins_x=params.num_bins_x,
            num_bins_y=params.num_bins_y,
            num_movable_nodes=num_movable_nodes_fence_region,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=num_filler_nodes_fence_region,
        )

        gl = greedy_legalize.GreedyLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=num_pins_in_nodes,
            flat_region_boxes=flat_region_boxes,
            flat_region_boxes_start=flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_bins_x=1,
            num_bins_y=64,
            # num_bins_x=64, num_bins_y=64,
            num_movable_nodes=num_movable_nodes_fence_region,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=num_filler_nodes_fence_region,
        )
        # for standard cell legalization
        al = abacus_legalize.AbacusLegalize(
            node_size_x=node_size_x,
            node_size_y=node_size_y,
            node_weights=num_pins_in_nodes,
            flat_region_boxes=flat_region_boxes,
            flat_region_boxes_start=flat_region_boxes_start,
            node2fence_region_map=node2fence_region_map,
            fp_info=data_collections.fp_info,
            num_bins_x=1,
            num_bins_y=64,
            num_movable_nodes=num_movable_nodes_fence_region,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=num_filler_nodes_fence_region,
        )

        def build_greedy_legalization_op(pos):
            ### reconstruct pos for fence region
            pos_total = pos.data.clone()
            pos = pos.view(2, -1)
            pos = (
                torch.cat(
                    [
                        pos[:, :num_movable_nodes][:, fence_region_mask],  ## movable
                        pos[
                            :, num_movable_nodes : num_movable_nodes + num_terminals
                        ],  ## terminals
                        virtual_macros_pos,  ## virtual macros
                        pos[
                            :,
                            num_movable_nodes
                            + num_terminals : num_movable_nodes
                            + num_terminals
                            + num_terminal_NIs,
                        ],  ## terminal NIs
                        pos[
                            :,
                            num_nodes
                            - num_filler_nodes
                            + filler_beg : num_nodes
                            - num_filler_nodes
                            + filler_end,
                        ],  ## fillers
                    ],
                    1,
                )
                .view(-1)
                .contiguous()
            )
            assert pos.size(0) == 2 * node_size_x.size(0)

            logging.info("Start legalization")
            pos1 = ml(pos, pos)
            result = gl(pos1, pos1)
            ## commit legal solution for movable cells in fence region
            pos_total = pos_total.view(2, -1)
            result = result.view(2, -1)
            pos_total[0, :num_movable_nodes].masked_scatter_(
                fence_region_mask, result[0, :num_movable_nodes_fence_region]
            )
            pos_total[1, :num_movable_nodes].masked_scatter_(
                fence_region_mask, result[1, :num_movable_nodes_fence_region]
            )
            pos_total = pos_total.view(-1).contiguous()
            result = result.view(-1).contiguous()
            return pos_total, pos1, result

        def build_abacus_legalization_op(pos_total, pos_ref, pos):
            result = al(pos_ref, pos)
            ### commit abacus results to pos_total
            pos_total = pos_total.view(2, -1)
            result = result.view(2, -1)
            pos_total[0, :num_movable_nodes].masked_scatter_(
                fence_region_mask, result[0, :num_movable_nodes_fence_region]
            )
            pos_total[1, :num_movable_nodes].masked_scatter_(
                fence_region_mask, result[1, :num_movable_nodes_fence_region]
            )
            pos_total = pos_total.view(-1).contiguous()
            return pos_total

        return build_greedy_legalization_op, build_abacus_legalization_op

    def build_detailed_placement(self, params, placedb, data_collections, device):
        """
        @brief detailed placement consisting of global swap and independent set matching
        @param params parameters
        @param placedb placement database
        @param data_collections a collection of all data and variables required for constructing the ops
        @param device cpu or cuda
        """
        gs = global_swap.GlobalSwap(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            fp_info=data_collections.fp_info,
            num_bins_x=placedb.num_bins_x // 2,
            num_bins_y=placedb.num_bins_y // 2,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            batch_size=256,
            max_iters=2,
            algorithm="concurrent",
        )
        kr = k_reorder.KReorder(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            fp_info=data_collections.fp_info,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            K=4,
            max_iters=2,
        )
        ism = independent_set_matching.IndependentSetMatching(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            flat_net2pin_map=data_collections.flat_net2pin_map,
            flat_net2pin_start_map=data_collections.flat_net2pin_start_map,
            pin2net_map=data_collections.pin2net_map,
            flat_node2pin_map=data_collections.flat_node2pin_map,
            flat_node2pin_start_map=data_collections.flat_node2pin_start_map,
            pin2node_map=data_collections.pin2node_map,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            net_mask=data_collections.net_mask_ignore_large_degrees,
            fp_info=data_collections.fp_info,
            num_bins_x=placedb.num_bins_x,
            num_bins_y=placedb.num_bins_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_terminal_NIs=placedb.num_terminal_NIs,
            num_filler_nodes=placedb.num_filler_nodes,
            batch_size=2048,
            set_size=128,
            max_iters=50,
            algorithm="concurrent",
        )

        # wirelength for position
        def build_detailed_placement_op(pos):
            logging.info("Start ABCDPlace for refinement")
            pos1 = pos
            legal = self.op_collections.legality_check_op(pos1)
            logging.info("ABCDPlace input legal flag = %d" % (legal))
            if not legal:
                return pos1

            # integer factorization to prime numbers
            def prime_factorization(num):
                lt = []
                while num != 1:
                    for i in range(2, int(num + 1)):
                        if num % i == 0:  # i is a prime factor
                            lt.append(i)
                            num = num / i  # get the quotient for further factorization
                            break
                return lt

            # compute the scale factor for detailed placement
            # as the algorithms prefer integer coordinate systems
            scale_factor = params.scale_factor
            if params.scale_factor != 1.0:
                inv_scale_factor = int(round(1.0 / params.scale_factor))
                prime_factors = prime_factorization(inv_scale_factor)
                target_inv_scale_factor = 1
                for factor in prime_factors:
                    if factor != 2 and factor != 5:
                        target_inv_scale_factor = inv_scale_factor
                        break
                scale_factor = 1.0 / target_inv_scale_factor
                logging.info(
                    "Deriving from system scale factor %g (1/%d)"
                    % (params.scale_factor, inv_scale_factor)
                )
                logging.info(
                    "Use scale factor %g (1/%d) for detailed placement"
                    % (scale_factor, target_inv_scale_factor)
                )

            for i in range(1):
                pos1 = kr(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1
                pos1 = ism(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("Independent set matching legal flag = %d" % (legal))
                if not legal:
                    return pos1
                pos1 = gs(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("Global swap legal flag = %d" % (legal))
                if not legal:
                    return pos1
                pos1 = kr(pos1, scale_factor)
                legal = self.op_collections.legality_check_op(pos1)
                logging.info("K-Reorder legal flag = %d" % (legal))
                if not legal:
                    return pos1
            return pos1

        return build_detailed_placement_op

    def build_draw_placement(self, params, placedb, data_collections):
        """
        @brief plot placement
        @param params parameters
        @param placedb placement database
        """
        return draw_place.DrawPlace(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            pin_offset_x=data_collections.pin_offset_x,
            pin_offset_y=data_collections.pin_offset_y,
            theta=data_collections.best_theta,
            pin2node_map=data_collections.pin2node_map,
            fp_info=data_collections.fp_info,
            bin_size_x=placedb.bin_size_x,
            bin_size_y=placedb.bin_size_y,
            num_movable_nodes=placedb.num_movable_nodes,
            num_filler_nodes=placedb.num_filler_nodes,
        )

    def validate(self, placedb, pos, iteration):
        """
        @brief validate placement
        @param placedb placement database
        @param pos locations of cells
        @param iteration optimization step
        """
        pos = torch.from_numpy(pos).to(self.device)
        hpwl = self.op_collections.hpwl_op(pos)
        overflow, max_density = self.op_collections.density_overflow_op(pos)

        return hpwl, overflow, max_density

    def plot(self, params, placedb, iteration, pos):
        """
        @brief plot layout
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        """
        tt = time.time()
        path = "%s/%s" % (params.result_dir, params.design_name())
        figname = "%s/plot/iter%s.png" % (path, "{:04}".format(iteration))
        os.system("mkdir -p %s" % (os.path.dirname(figname)))
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)
        self.op_collections.draw_place_op(pos, figname)
        logging.info("plotting to %s takes %.3f seconds" % (figname, time.time() - tt))

    def dump(self, params, placedb, pos, filename):
        """
        @brief dump intermediate solution as compressed pickle file (.pklz)
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        @param filename output file name
        """
        with gzip.open(filename, "wb") as f:
            pickle.dump(
                (
                    self.data_collections.node_size_x.cpu(),
                    self.data_collections.node_size_y.cpu(),
                    self.data_collections.flat_net2pin_map.cpu(),
                    self.data_collections.flat_net2pin_start_map.cpu(),
                    self.data_collections.pin2net_map.cpu(),
                    self.data_collections.flat_node2pin_map.cpu(),
                    self.data_collections.flat_node2pin_start_map.cpu(),
                    self.data_collections.pin2node_map.cpu(),
                    self.data_collections.pin_offset_x.cpu(),
                    self.data_collections.pin_offset_y.cpu(),
                    self.data_collections.net_mask_ignore_large_degrees.cpu(),
                    self.data_collections.fp_info.xl,
                    self.data_collections.fp_info.yl,
                    self.data_collections.fp_info.xh,
                    self.data_collections.fp_info.yh,
                    self.data_collections.fp_info.site_width,
                    self.data_collections.fp_info.row_height,
                    placedb.num_bins_x,
                    placedb.num_bins_y,
                    placedb.num_movable_nodes,
                    placedb.num_terminal_NIs,
                    placedb.num_filler_nodes,
                    pos,
                ),
                f,
            )

    def load(self, params, placedb, filename):
        """
        @brief dump intermediate solution as compressed pickle file (.pklz)
        @param params parameters
        @param placedb placement database
        @param iteration optimization step
        @param pos locations of cells
        @param filename output file name
        """
        with gzip.open(filename, "rb") as f:
            data = pickle.load(f)
            self.data_collections.node_size_x.data = data[0].data.to(self.device)
            self.data_collections.node_size_y.data = data[1].data.to(self.device)
            self.data_collections.flat_net2pin_map.data = data[2].data.to(self.device)
            self.data_collections.flat_net2pin_start_map.data = data[3].data.to(
                self.device
            )
            self.data_collections.pin2net_map.data = data[4].data.to(self.device)
            self.data_collections.flat_node2pin_map.data = data[5].data.to(self.device)
            self.data_collections.flat_node2pin_start_map.data = data[6].data.to(
                self.device
            )
            self.data_collections.pin2node_map.data = data[7].data.to(self.device)
            self.data_collections.pin_offset_x.data = data[8].data.to(self.device)
            self.data_collections.pin_offset_y.data = data[9].data.to(self.device)
            self.data_collections.net_mask_ignore_large_degrees.data = data[10].data.to(
                self.device
            )

            placedb.xl = data[11]
            placedb.yl = data[12]
            placedb.xh = data[13]
            placedb.yh = data[14]
            placedb.site_width = data[15]
            placedb.row_height = data[16]
            placedb.num_bins_x = data[17]
            placedb.num_bins_y = data[18]
            num_movable_nodes = data[19]
            num_nodes = data[0].numel()
            placedb.num_terminal_NIs = data[20]
            placedb.num_filler_nodes = data[21]
            placedb.num_physical_nodes = num_nodes - placedb.num_filler_nodes
            placedb.num_terminals = (
                placedb.num_physical_nodes
                - placedb.num_terminal_NIs
                - num_movable_nodes
            )
            self.data_collections.pos[0].data = data[22].data.to(self.device)
