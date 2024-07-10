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
int computeNetCrossingCudaLauncher<T>(
    const T* x, const T* y, const int* flat_netpin,
    const int* netpin_start, const unsigned char* net_mask,
    int num_nets, T* net_crossing,
    T *lambda_, T *mu_, T *sigma_,
    T *grad_intermediate_x, T *grad_intermediate_y
    );

DREAMPLACE_END_NAMESPACE