/**
 * @file   net_crossing.cpp
 * @author Niansong Zhang
 * @date   Jul 2024
 * @brief  Compute net crossing on CPU
 */
#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE


template <typename T>
void computeNetCrossingLauncher(const T* x, const T* y, const int* flat_netpin,
                               const int* netpin_start, const unsigned char* net_mask,
                               int num_nets, T* net_crossing,
                               T *grad_intermediate_x, T *grad_intermediate_y,
                               int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nets; ++i) {
    for (int j = 0; j < num_nets; ++j) {
      if (i == j) continue;
      if (!net_mask[i] || !net_mask[j]) continue;

      // TODO(Niansong): if j > i directly read from previous computation results

      // skip if net i/j has only one pin
      if (netpin_start[i + 1] - netpin_start[i] <= 1 || netpin_start[j + 1] - netpin_start[j] <= 1) {
        continue;
      }
      for (int net_i_sink_pin_idx = netpin_start[i] + 1; net_i_sink_pin_idx < netpin_start[i + 1]; ++net_i_sink_pin_idx) {
        int net_i_src_pin_id = flat_netpin[netpin_start[i]];
        int net_i_sink_pin_id = flat_netpin[net_i_sink_pin_idx];
        for (int net_j_sink_pin_idx = netpin_start[j] + 1; net_j_sink_pin_idx < netpin_start[j + 1]; ++net_j_sink_pin_idx) {
          int net_j_src_pin_id = flat_netpin[netpin_start[j]];
          int net_j_sink_pin_id = flat_netpin[net_j_sink_pin_idx];
          
          T x1 = x[net_i_src_pin_id];
          T y1 = y[net_i_src_pin_id];
          T x2 = x[net_i_sink_pin_id];
          T y2 = y[net_i_sink_pin_id];
          T x3 = x[net_j_src_pin_id];
          T y3 = y[net_j_src_pin_id];
          T x4 = x[net_j_sink_pin_id];
          T y4 = y[net_j_sink_pin_id];

           // Bezier curve intersection
          T r = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
          T u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
          u = -u;

          // Bell function
          // TODO(Niansong): parameterize
          auto bell = [](T x) {
            if std::abs(x) <= 0.5 {
              return 1 - 2 * x * x;
            } else if std::abs(x) <= 1 {
              return 2 * (1 - std::abs(x)) * (1 - std::abs(x));
            } else {
              return 0.0;
            }
          };

          net_crossing[i] += bell(r - 0.5) * bell(u - 0.5);
        }
      }
    }
  }


}

/// @brief Compute net crossing score
/// @param pos cell locations, array of x locations and then y locations
/// @param flat_netpin similar to the JA array in CSR format, which is flattened
/// from the net2pin map (array of array)
/// @param netpin_start similar to the IA array in CSR format, IA[i+1]-IA[i] is
/// the number of pins in each net, the length of IA is number of nets + 1
/// @param net_mask an array to record whether compute the netcrossing for a net or
/// not
at::Tensor net_crossing_forward(at::Tensor pos, at::Tensor flat_netpin,
                               at::Tensor netpin_start, at::Tensor net_mask) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CPU(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CPU(net_mask);
  CHECK_CONTIGUOUS(net_mask);


  int num_nets = netpin_start.numel() - 1;
  at::Tensor net_crossing = at::zeros(num_nets, pos.options());

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeNetCrossingLauncher", [&] {
    computeNetCrossingLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + pos.numel() / 2,
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), num_nets,
        at::get_num_threads(), DREAMPLACE_TENSOR_DATA_PTR(net_crossing, scalar_t));
  });

  return net_crossing;
}



DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::net_crossing_forward,
        "NetCrossing forward");
}
