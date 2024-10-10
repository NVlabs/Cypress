/**
 * @file net_crossing_cuda.cpp
 * @author Niansong Zhang
 * @date Jul 2024
 * @brief CUDA implementation of net crossing counting
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
int computeNetCrossingCudaLauncher(
    const T* x, const T* y, const int* flat_netpin,
    const int* netpin_start, const unsigned char* net_mask,
    int num_nets, int num_pins, T* net_crossing,
    T *lambda_, T *mu_, T *sigma_,
    T *grad_intermediate_x, T *grad_intermediate_y
);

std::vector<at::Tensor> net_crossing_forward(at::Tensor pos, at::Tensor flat_netpin,
                               at::Tensor netpin_start, at::Tensor net_mask,
                               at::Tensor lambda, at::Tensor mu, at::Tensor sigma // scalar Tensors
) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);
  CHECK_FLAT_CUDA(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);
  CHECK_FLAT_CUDA(net_mask);
  CHECK_CONTIGUOUS(net_mask);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pos.numel() / 2;

  at::Tensor net_crossing = at::zeros(num_nets, pos.options());
  at::Tensor grad_intermediate = at::zeros_like(pos);

  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "computeNetCrossingCudaLauncher", [&] {
    computeNetCrossingCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pos, scalar_t) + num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(net_mask, unsigned char), 
        num_nets, num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(net_crossing, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(lambda, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(mu, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(sigma, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_intermediate, scalar_t) + num_pins);
  });

  // Check if there are any NaN values in the tensor
  bool has_nan = at::isnan(net_crossing).any().item<bool>();
  if (has_nan){
    std::cout << "net crossing contains NaN: " << (has_nan ? "Yes" : "No") << std::endl;
  }

  return {net_crossing.sum(), grad_intermediate};
}

at::Tensor net_crossing_backward(
  at::Tensor grad_pos, at::Tensor pos, at::Tensor grad_intermediate
) {

  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(grad_intermediate);
  CHECK_EVEN(grad_intermediate);
  CHECK_CONTIGUOUS(grad_intermediate);

  at::Tensor grad_out = grad_intermediate.mul_(grad_pos);

  return grad_out;
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::net_crossing_forward, "NetCrossing forward (CUDA)");
  m.def("backward", &DREAMPLACE_NAMESPACE::net_crossing_backward, "NetCrossing backward (CUDA)");
}