/*
* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/**
 * @file   draw_place.h
 * @author Yibo Lin
 * @date   Jan 2019
 */
#ifndef DREAMPLACE_DRAW_PLACE_H
#define DREAMPLACE_DRAW_PLACE_H

#include "draw_place/src/PlaceDrawer.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int drawPlaceLauncher(const T* x_tensor, const T* y_tensor,
                      const T* node_size_x_tensor, const T* node_size_y_tensor,
                      const T* pin_offset_x_tensor,
                      const T* pin_offset_y_tensor,
                      const T* theta,
                      const int* side,
                      const int* pin2node_map_tensor, const int num_nodes,
                      const int num_movable_nodes, const int num_filler_nodes,
                      const int num_pins, const T xl, const T yl, const T xh,
                      const T yh, const T site_width, const T row_height,
                      const T bin_size_x, const T bin_size_y,
                      const std::string& filename, const bool show_fillers) {
  PlaceDrawer<T, int> drawer(
      x_tensor, y_tensor, node_size_x_tensor, node_size_y_tensor,
      pin_offset_x_tensor, pin_offset_y_tensor, theta, side, pin2node_map_tensor, num_nodes,
      num_movable_nodes, num_filler_nodes, num_pins, xl, yl, xh, yh, site_width,
      row_height, bin_size_x, bin_size_y, show_fillers);
  typename PlaceDrawer<T, int>::FileFormat ff;
  if (filename.substr(filename.size() - 4) == ".eps") {
    ff = PlaceDrawer<T, int>::EPS;
  } else if (filename.substr(filename.size() - 4) == ".pdf") {
    ff = PlaceDrawer<T, int>::PDF;
  } else if (filename.substr(filename.size() - 4) == ".svg") {
    ff = PlaceDrawer<T, int>::SVG;
  } else if (filename.substr(filename.size() - 4) == ".png") {
    ff = PlaceDrawer<T, int>::PNG;
  } else {
    ff = PlaceDrawer<T, int>::GDSII;
  }
  return drawer.run(filename, ff);
}

DREAMPLACE_END_NAMESPACE

#endif


