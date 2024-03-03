#ifndef CUDA_RENDERING_FORWARD_H_INCLUDED
#define CUDA_RENDERING_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdint.h"

namespace FORWARD
{
	void preprocess(
		//input
		int N,
		const float* points,
		const int W, int H, 
		const float radius,
		const dim3 grid,
		//output
		int* radii,
		float* depths,
		float2* points_xy_image,
		uint32_t* tiles_touched);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const int* radii,
		float* final_T,
		uint32_t* n_contrib,
		float* out_color);
}


#endif