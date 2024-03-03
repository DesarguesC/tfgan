#ifndef CUDA_RENDERING_AUXILIARY_H_INCLUDED
#define CUDA_RENDERING_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - radius - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - radius - 1) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + radius + BLOCK_X) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + radius + BLOCK_Y) / BLOCK_Y)))
	};
}

__forceinline__ __device__ bool in_frustum(int idx, float3 point)
{
	if (point.z <= 0.2f)
	{
		return false;
	}
	return true;
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif