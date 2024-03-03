#ifndef CUDA_RENDERING_BACKWARD_H_INCLUDED
#define CUDA_RENDERING_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdint.h"

namespace BACKWARD
{
	void render(
		//input
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, float radius,
		const float2* points_xy_image,
		const float* depths,
		const float* features,
		const int* radii,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		//output
		float3* dL_dcoordinates,
		float* dL_dfeatures);
}

#endif