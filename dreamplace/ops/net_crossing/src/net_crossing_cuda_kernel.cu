/**
 * @file   net_crossing_cuda_kernel.cu
 * @author Niansong Zhang
 * @date   Jul 2024
 */

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T> // choose diff bell functions
__global__ void computeNetCrossing(
        const T* x, const T* y, 
        const int* flat_netpin, 
        const int* netpin_start, 
        const unsigned char* net_mask, 
        int num_nets, 
        T* net_crossing, 
        T *lambda_, T *mu_, T *sigma_,
        T *grad_intermediate_x, T *grad_intermediate_y
        )
{
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_nets * num_nets;
    if (thread_idx >= total_pairs) return; // boundary check
    int i = thread_idx / num_nets;
    int j = thread_idx % num_nets;
    if (j <= i) return; // only compute the upper triangular part
    if (!net_mask[i] || !net_mask[j]) return; // skip masked nets
    // skip if net i/j has only one pin
    if (netpin_start[i+1] - netpin_start[i] <= 1 || netpin_start[j+1] - netpin_start[j] <= 1) return;
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
            T df_dx1 = bell_gradient(t - 0.5) * dt_dx1;
            T df_dy1 = bell_gradient(t - 0.5) * dt_dy1;
            T df_dx2 = bell_gradient(t - 0.5) * dt_dx2;
            T df_dy2 = bell_gradient(t - 0.5) * dt_dy2;
            T df_dx3 = bell_gradient(t - 0.5) * dt_dx3;
            T df_dy3 = bell_gradient(t - 0.5) * dt_dy3;
            T df_dx4 = bell_gradient(t - 0.5) * dt_dx4;
            T df_dy4 = bell_gradient(t - 0.5) * dt_dy4;

            T dg_dx1 = bell_gradient(u - 0.5) * du_dx1;
            T dg_dy1 = bell_gradient(u - 0.5) * du_dy1;
            T dg_dx2 = bell_gradient(u - 0.5) * du_dx2;
            T dg_dy2 = bell_gradient(u - 0.5) * du_dy2;
            T dg_dx3 = bell_gradient(u - 0.5) * du_dx3;
            T dg_dy3 = bell_gradient(u - 0.5) * du_dy3;
            T dg_dx4 = bell_gradient(u - 0.5) * du_dx4;
            T dg_dy4 = bell_gradient(u - 0.5) * du_dy4;

            T dx1 = df_dx1 * bell(u - 0.5) + bell(t - 0.5) * dg_dx1;
            T dy1 = df_dy1 * bell(u - 0.5) + bell(t - 0.5) * dg_dy1;
            T dx2 = df_dx2 * bell(u - 0.5) + bell(t - 0.5) * dg_dx2;
            T dy2 = df_dy2 * bell(u - 0.5) + bell(t - 0.5) * dg_dy2;
            T dx3 = df_dx3 * bell(u - 0.5) + bell(t - 0.5) * dg_dx3;
            T dy3 = df_dy3 * bell(u - 0.5) + bell(t - 0.5) * dg_dy3;
            T dx4 = df_dx4 * bell(u - 0.5) + bell(t - 0.5) * dg_dx4;
            T dy4 = df_dy4 * bell(u - 0.5) + bell(t - 0.5) * dg_dy4;
          
            atomicAdd(grad_intermediate_x + net_i_src_pin_id, dx1);
            atomicAdd(grad_intermediate_y + net_i_src_pin_id, dy1);
            atomicAdd(grad_intermediate_x + net_i_sink_pin_id, dx2);
            atomicAdd(grad_intermediate_y + net_i_sink_pin_id, dy2);
            atomicAdd(grad_intermediate_x + net_j_src_pin_id, dx3);
            atomicAdd(grad_intermediate_y + net_j_src_pin_id, dy3);
            atomicAdd(grad_intermediate_x + net_j_sink_pin_id, dx4);
            atomicAdd(grad_intermediate_y + net_j_sink_pin_id, dy4);
        }
    }
}


template <typename T>
int computeNetCrossingCudaLauncher(
        const T* x, const T* y, const int* flat_netpin,
        const int* netpin_start, const unsigned char* net_mask,
        int num_nets, T* net_crossing,
        T *lambda_, T *mu_, T *sigma_,
        T *grad_intermediate_x, T *grad_intermediate_y
        )
{
    int thread_count = 256; 
    int total_pairs = num_nets * num_nets;
    // int total_pairs = num_net * (num_nets - 1) / 2; // TODO: use this to avoid launching more than needed
    int block_count = ceilDiv(total_pairs, thread_count);

    computeNetCrossing<<<block_count, thread_count>>>(
            x, y,
            flat_netpin,
            netpin_start,
            net_mask,
            num_nets,
            net_crossing,
            lambda_, mu_, sigma_,
            grad_intermediate_x, grad_intermediate_y
            );

    return 0; 
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computeNetCrossingCudaLauncher<T>(\
            const T* x, const T* y, const int* flat_netpin, \
            const int* netpin_start, const unsigned char* net_mask, \
            int num_nets, T* net_crossing, \
            T *lambda_, T *mu_, T *sigma_, \
            T *grad_intermediate_x, T *grad_intermediate_y \
            );

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
