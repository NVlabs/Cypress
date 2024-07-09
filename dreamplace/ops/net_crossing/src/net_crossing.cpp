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
                               T *lambda_, T *mu_, T *sigma_,
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
          T t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
          T u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));

          // std::cout << "custom t: " << t << " u: " << u << "\n";

          // Bell function
          // lambda = 2, mu = 2, sigma = 1
          T lambda = lambda_[0];
          T mu = mu_[0];
          T sigma = sigma_[0];
          auto bell = [&](T x) {
            if (std::abs(x) <= 0.5) {
              return 1 - lambda * x * x;
            } else if (std::abs(x) <= 1) {
              return mu * (sigma - std::abs(x)) * (sigma - std::abs(x));
            } else {
              return (T) 0.0;
            }
          };

          net_crossing[i] += bell(t - 0.5) * bell(u - 0.5);

          // compute gradient
          auto bell_gradient = [&](T x) {
            if (std::abs(x) <= 0.5) {
              return -2 * lambda * x;
            } else if (std::abs(x) <= 1) {
              return 2 * mu * x * (std::abs(x) - sigma) / std::abs(x);
            } else {
              return (T) 0.0;
            }
          };

          T dt_dx1 = ((y3 - y4) * (x4 * (y3 - y2) + x3 * (y2 - y4) + x2 * (y4 - y3))) / std::pow(-x3 * (y1 - y2) - x4 * (y2 - y1) + (x1 - x2) * (y3 - y4), 2);
          T dt_dy1 = ((x3 - x4) * (-x4 * (y3 - y2) - x3 * (y2 - y4) + x2 * (y3 - y4))) / std::pow((x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2), 2);
          T dt_dx2 = -((y4 - y3) * ((x1 - x3) * (y3 - y4) - (x3 - x4) * (y1 - y3))) / std::pow((x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2), 2);
          T dt_dy2 = -((x3 - x4) * ((x1 - x3) * (y3 - y4) - (x3 - x4) * (y1 - y3))) / std::pow((x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2), 2);
          T dt_dx3 = ((y4 - y3) * (-x4 * (y2 - y1) - x2 * (y1 - y4) + x1 * (y2 - y4))) / std::pow((x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2), 2);
          T dt_dy3 = ((x3 - x4) * (x4 * (y1 - y2) + x1 * (y2 - y4) + x2 * (y4 - y1))) / std::pow(-x3 * (y1 - y2) - x4 * (y2 - y1) + (x1 - x2) * (y3 - y4), 2);
          T dt_dx4 = ((y3 - y4) * (x3 * (y1 - y2) + x1 * (y2 - y3) + x2 * (y3 - y1))) / std::pow((x3 - x4) * (y1 - y2) - (x1 - x2) * (y3 - y4), 2);
          T dt_dy4 = ((x4 - x3) * (-x3 * (y2 - y1) - x2 * (y1 - y3) + x1 * (y2 - y3)))/ std::pow((x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2), 2);

          T du_dx1 = ((y2 - y1) * (-x4 * (y3 - y2) - x3 * (y2 - y4) + x2 * (y3 - y4))) / std::pow((x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2), 2);
          T du_dy1 = ((x1 - x2) * (x4 * (y2 - y3) + x2 * (y3 - y4) + x3 * (y4 - y2))) / std::pow(-x3 * (y1 - y2) - x4 * (y2 - y1) + (x1 - x2) * (y3 - y4), 2);
          T du_dx2 = -((y1 - y2) * (x4 * (y3 - y1) + x3 * (y1 - y4) + x1 * (y4 - y3))) / std::pow(-x3 * (y1 - y2) - x4 * (y2 - y1) + (x1 - x2) * (y3 - y4), 2);
          T du_dy2 = -((x1 - x2) * (x4 * (y1 - y3) + x1 * (y3 - y4) + x3 * (y4 - y1))) / std::pow((x3 - x4) * (y1 - y2) - (x1 - x2) * (y3 - y4), 2);
          T du_dx3 = ((y1 - y2) * (x4 * (y2 - y1) + x2 * (y1 - y4) + x1 * (y4 - y2))) / std::pow(-x3 * (y1 - y2) - x4 * (y2 - y1) + (x1 - x2) * (y3 - y4), 2);
          T du_dy3 = ((x1 - x2) * (x4 * (y1 - y2) + x1 * (y2 - y4) + x2 * (y4 - y1))) / std::pow(-x3 * (y1 - y2) - x4 * (y2 - y1) + (x1 - x2) * (y3 - y4), 2);
          T du_dx4 = ((y1 - y2) * ((x1 - x2) * (y1 - y3) - (x1 - x3) * (y1 - y2))) / std::pow((x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2), 2);
          T du_dy4 = ((x2 - x1) * ((x1 - x2) * (y1 - y3) - (x1 - x3) * (y1 - y2))) / std::pow((x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2), 2);

          // NC = f * g
          T df_dx1 = bell_gradient(t) * dt_dx1;
          T df_dy1 = bell_gradient(t) * dt_dy1;
          T df_dx2 = bell_gradient(t) * dt_dx2;
          T df_dy2 = bell_gradient(t) * dt_dy2;
          T df_dx3 = bell_gradient(t) * dt_dx3;
          T df_dy3 = bell_gradient(t) * dt_dy3;
          T df_dx4 = bell_gradient(t) * dt_dx4;
          T df_dy4 = bell_gradient(t) * dt_dy4;

          T dg_dx1 = bell_gradient(u) * du_dx1;
          T dg_dy1 = bell_gradient(u) * du_dy1;
          T dg_dx2 = bell_gradient(u) * du_dx2;
          T dg_dy2 = bell_gradient(u) * du_dy2;
          T dg_dx3 = bell_gradient(u) * du_dx3;
          T dg_dy3 = bell_gradient(u) * du_dy3;
          T dg_dx4 = bell_gradient(u) * du_dx4;
          T dg_dy4 = bell_gradient(u) * du_dy4;

          T dx1 = df_dx1 * bell(u) + bell(t) * dg_dx1;
          T dy1 = df_dy1 * bell(u) + bell(t) * dg_dy1;
          T dx2 = df_dx2 * bell(u) + bell(t) * dg_dx2;
          T dy2 = df_dy2 * bell(u) + bell(t) * dg_dy2;
          T dx3 = df_dx3 * bell(u) + bell(t) * dg_dx3;
          T dy3 = df_dy3 * bell(u) + bell(t) * dg_dy3;
          T dx4 = df_dx4 * bell(u) + bell(t) * dg_dx4;
          T dy4 = df_dy4 * bell(u) + bell(t) * dg_dy4;
          
          grad_intermediate_x[net_i_src_pin_id] = dx1;
          grad_intermediate_y[net_i_src_pin_id] = dy1;
          grad_intermediate_x[net_i_sink_pin_id] = dx2;
          grad_intermediate_y[net_i_sink_pin_id] = dy2;
          grad_intermediate_x[net_j_src_pin_id] = dx3;
          grad_intermediate_y[net_j_src_pin_id] = dy3;
          grad_intermediate_x[net_j_sink_pin_id] = dx4;
          grad_intermediate_y[net_j_sink_pin_id] = dy4;
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
std::vector<at::Tensor> net_crossing_forward(at::Tensor pos, at::Tensor flat_netpin,
                               at::Tensor netpin_start, at::Tensor net_mask,
                               at::Tensor lambda, at::Tensor mu, at::Tensor sigma // scalar Tensors
) {
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
  int num_pins = pos.numel() / 2;

  at::Tensor net_crossing = at::zeros(num_nets, pos.options());
  at::Tensor grad_intermediate = at::zeros_like(pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeNetCrossingLauncher", [&] {
    computeNetCrossingLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), 
        num_nets,
        DREAMPLACE_TENSOR_DATA_PTR(net_crossing, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(lambda, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(mu, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(sigma, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins,
        at::get_num_threads());
  });

  return {net_crossing.sum(), grad_intermediate};
}


at::Tensor net_crossing_backward(
  at::Tensor grad_pos, at::Tensor pos, at::Tensor grad_intermediate
) {

  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(grad_intermediate);
  CHECK_EVEN(grad_intermediate);
  CHECK_CONTIGUOUS(grad_intermediate);

  at::Tensor grad_out = grad_intermediate.mul_(grad_pos);

  return grad_out;
}


DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::net_crossing_forward,
        "NetCrossing forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::net_crossing_backward,
        "NetCrossing backward");
}
