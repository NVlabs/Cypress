#include <cfloat>
#include <stdio.h>
#include "assert.h"
#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

/// @brief Compute pin position from node position 
template <typename T, typename K>
__global__ void computePinPos(
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const T* theta,
	const K* pin2node_map,
	const int num_pins,
	const T* h,
	const T* w,
	T* pin_x, T* pin_y
	)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_pins)
	{
		int node_id = pin2node_map[i];

		// orignal version		
		// pin_x[i] = pin_offset_x[i] + x[node_id];
		// pin_y[i] = pin_offset_y[i] + y[node_id];

		// with rotation
		T xi = x[node_id];
		T yi = y[node_id];
		T hi = h[node_id];
		T wi = w[node_id];
		T cos_theta = std::cos(theta[node_id]);
		T sin_theta = std::sin(theta[node_id]);
		T ofx = pin_offset_x[i];
		T ofy = pin_offset_y[i];
		pin_x[i] = (xi + wi/2) + (ofx - wi/2) * cos_theta - (ofy - hi/2) * sin_theta;
		pin_y[i] = (yi + hi/2) + (ofx - wi/2) * sin_theta + (ofy - hi/2) * cos_theta;
	}
}

template <typename T>
int computePinPosCudaLauncher(
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const T* theta,
	const long* pin2node_map,
	const int* flat_node2pin_map,
	const int* flat_node2pin_start_map,
	int num_pins,
	const T* h, 
	const T* w,
	T* pin_x, T* pin_y
    )
{
	int thread_count = 512;

	computePinPos<<<(num_pins+thread_count-1) / thread_count, thread_count>>>(x, y, pin_offset_x, pin_offset_y, theta, pin2node_map, num_pins, h, w, pin_x, pin_y);

    return 0;
}

/// @brief Compute pin position from node position 
template <typename T>
__global__ void computeNodeGrad(
	const T* grad_out_x,
	const T* grad_out_y, // pin grad back propagated from downstream ops
	const int* flat_node2pin_map,
    const int* flat_node2pin_start_map, 
	const T* theta,
	const T* pin_offset_x,
	const T* pin_offset_y,
    const int num_nodes, 
	const T* h,
	const T* w,
	T* grad_x,
	T* grad_y,
	T* grad_theta
	)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes)
	{
        T& gx = grad_x[i];
        T& gy = grad_y[i];
		T& gt = grad_theta[i];
        gx = 0; 
        gy = 0; 
		gt = 0;
		T cos_theta = std::cos(theta[i]);
		T sin_theta = std::sin(theta[i]);
		T hi = h[i];
		T wi = w[i];
        for (int j = flat_node2pin_start_map[i]; j < flat_node2pin_start_map[i+1]; ++j)
        {
            int pin_id = flat_node2pin_map[j]; 
            gx += grad_out_x[pin_id]; 
            gy += grad_out_y[pin_id]; 
			T px_pt = cos_theta * (hi/2 - pin_offset_y[pin_id]) + sin_theta * (wi/2 - pin_offset_x[pin_id]);
			T py_pt = cos_theta * (pin_offset_x[pin_id] - wi/2) + sin_theta * (pin_offset_y[pin_id] - hi/2);
			gt += grad_out_x[pin_id] * px_pt + grad_out_y[pin_id] * py_pt;
        }
	}
}

template <typename T>
int computePinPosGradCudaLauncher(
	const T* grad_out_x, const T* grad_out_y,
	const T* x, const T* y,
	const T* pin_offset_x,
	const T* pin_offset_y,
	const T* theta,
	const long* pin2node_map,
	const int* flat_node2pin_map,
	const int* flat_node2pin_start_map,
	const T* h, const T* w,
	int num_nodes,
	int num_pins,
	T* grad_x, T* grad_y,
	T* grad_theta
    )
{
    int thread_count = 512;

    computeNodeGrad<<<(num_nodes + thread_count - 1) / thread_count, thread_count>>>(
            grad_out_x, 
            grad_out_y, 
            flat_node2pin_map, 
            flat_node2pin_start_map,
			theta,
			pin_offset_x,
			pin_offset_y,
            num_nodes, 
			h,
			w,
            grad_x, 
            grad_y,
			grad_theta
            );

    return 0;	
}


#define REGISTER_KERNEL_LAUNCHER(T) \
    template int computePinPosCudaLauncher<T>(\
    	    const T* x, const T* y, \
    	    const T* pin_offset_x, \
	        const T* pin_offset_y, \
			const T* theta, \
	        const long* pin2node_map, \
	        const int* flat_node2pin_map, \
	        const int* flat_node2pin_start_map, \
	        int num_pins, \
			const T* h, const T* w, \
	        T* pin_x, T* pin_y \
            ); \
    \
    template int computePinPosGradCudaLauncher<T>(\
        	const T* grad_out_x, const T* grad_out_y, \
	        const T* x, const T* y, \
	        const T* pin_offset_x, \
	        const T* pin_offset_y, \
			const T* theta, \
	        const long* pin2node_map, \
	        const int* flat_node2pin_map, \
	        const int* flat_node2pin_start_map, \
			const T* h, const T* w, \
	        int num_nodes, \
	        int num_pins, \
	        T* grad_x, T* grad_y, \
			T* grad_theta \
            ); 

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
