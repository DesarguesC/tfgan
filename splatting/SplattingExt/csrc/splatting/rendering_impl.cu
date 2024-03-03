#include "rendering_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}


__global__ void duplicateWithKeys(
	int P, //高斯点的数量。
	const float2* points_xy, //每个高斯点在屏幕空间的 x,y 坐标数组。
	const float* depths, //每个高斯点在视图空间的深度数组。
	const uint32_t* offsets, //用于确定每个高斯点键/值对开始写入位置的偏移数组。
	uint64_t* voxel_keys_unsorted, //用于存储生成的未排序键的数组。
	uint32_t* voxel_values_unsorted, //用于存储对应的高斯点索引作为值的数组。
	int* radii, //每个高斯点的屏幕空间半径。
	dim3 grid) //定义了CUDA网格的维度，通常与屏幕空间的图块对应。
{
	auto idx = cg::this_grid().thread_rank(); //获取当前CUDA线程的全局索引 idx。
	if (idx >= P)
		return;

	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1]; //对于那些屏幕空间半径大于0（可见）的高斯点，计算出它们在输出缓冲区中的偏移量 off。如果是第一个高斯点，则偏移量为0；否则，偏移量为前一个高斯点的偏移值。
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid); //计算当前高斯点覆盖的屏幕空间矩形的最小和最大坐标 (rect_min, rect_max)。
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				voxel_keys_unsorted[off] = key;
				voxel_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32; 
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

CudaRendering::GeometryState CudaRendering::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.radii, P, 128);
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.points_xy_image, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P); 
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRendering::ImageState CudaRendering::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRendering::BinningState CudaRendering::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

int CudaRendering::Rendering::forward(
	std::function<char* (size_t)> geometryBuffer, 
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int N, 
	const int width, int height,
	const float radius,
	const float* coordinates,
	const float* features,
	float* out_color
)
{

	size_t chunk_size = required<GeometryState>(N);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, N);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	CHECK_CUDA(FORWARD::preprocess(
		N, 
		coordinates,
		width, height,
		radius,
		tile_grid,
		geomState.radii,
		geomState.depths,
		geomState.points_xy_image,
		geomState.tiles_touched
	), false)

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, N), false)

	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost), false);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	duplicateWithKeys << <(N + 255) / 256, 256 >> > (
		N,
		geomState.points_xy_image,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		geomState.radii,
		tile_grid);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), false)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), false);

	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);

	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
        //input
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.points_xy_image,
		features,
        geomState.radii,
        //output
		imgState.accum_alpha,
		imgState.n_contrib,
		out_color), false);

	return num_rendered;
}

void CudaRendering::Rendering::backward(
    //input
	const int N, int num_rendered,
	const int width, int height, float radius,
	const float* coordinates,
	const float* features,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
    //output
	float* dL_dcoordinates,
	float* dL_dfeatures)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, N);
	BinningState binningState = BinningState::fromChunk(binning_buffer, num_rendered);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	CHECK_CUDA(BACKWARD::render(
		//input
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, radius,
		geomState.points_xy_image,
        geomState.depths,
		features,
        geomState.radii,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		//output
        (float3*)dL_dcoordinates,
		dL_dfeatures), false);
}