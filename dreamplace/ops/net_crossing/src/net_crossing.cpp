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
int computeNetCrossingLauncher(const T* x, const T* y, const int* flat_netpin,
                               const int* netpin_start, const unsigned char* net_mask,
                               int num_nets, int num_threads, T* net_crossing){
#pragma omp parallel for num_threads(num_threads)
  for (int i = 0; i < num_nets; ++i) {
    for (int j = 0; j < num_nets; ++j) {
      if (i == j) continue;
      if (!net_mask[i] || !net_mask[j]) continue;

      // find the bounding box of pins in net i and net j
      T i_max_x = -std::numeric_limits<T>::max();
      T i_min_x = std::numeric_limits<T>::max();
      T i_max_y = -std::numeric_limits<T>::max();
      T i_min_y = std::numeric_limits<T>::max();
      T j_max_x = -std::numeric_limits<T>::max();
      T j_min_x = std::numeric_limits<T>::max();
      T j_max_y = -std::numeric_limits<T>::max();
      T j_min_y = std::numeric_limits<T>::max();
      for (int k = netpin_start[i]; k < netpin_start[i + 1]; ++k) {
        int pin_id_i = flat_netpin[k];
        i_max_x = std::max(i_max_x, x[pin_id_i]);
        i_min_x = std::min(i_min_x, x[pin_id_i]);
        i_max_y = std::max(i_max_y, y[pin_id_i]);
        i_min_y = std::min(i_min_y, y[pin_id_i]);
      }
      for (int k = netpin_start[j]; k < netpin_start[j + 1]; ++k) {
        int pin_id_j = flat_netpin[k];
        j_max_x = std::max(j_max_x, x[pin_id_j]);
        j_min_x = std::min(j_min_x, x[pin_id_j]);
        j_max_y = std::max(j_max_y, y[pin_id_j]);
        j_min_y = std::min(j_min_y, y[pin_id_j]);
      }

      // net i: (i_min_x, i_min_y) -> (i_max_x, i_max_y)
      // net j: (j_min_x, j_min_y) -> (j_max_x, j_max_y)
      T x1 = i_min_x;
      T y1 = i_min_y;
      T x2 = i_max_x;
      T y2 = i_max_y;
      T x3 = j_min_x;
      T y3 = j_min_y;
      T x4 = j_max_x;
      T y4 = j_max_y;

      // Bezier curve intersection
      T r = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
      T u = (x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
      u = -u;

      // Bell function
      // TODO(Niansong): parameterize
      auto bell = [](T x) {
        if std::abs(x) <= 0.5 {
          return 0.75 - x * x;
        } else if std::abs(x) <= 1 {
          return 0.5 * (1.5 - std::abs(x)) * (1.5 - std::abs(x));
        } else {
          return 0.0;
        }
      };

      net_crossing[i] += bell(r - 0.5) * bell(u - 0.5);
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
