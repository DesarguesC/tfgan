#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template<int C>
__global__ void preprocessCUDA(
	//input
	int N,
	const float* coordinates,
	const int W, int H, 
	const float radius,
	const dim3 grid,
	//output
	int* radii,
	float* depths,
	float2* points_xy_image,
	uint32_t* tiles_touched
)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
		return;

	radii[idx] = 0;
	tiles_touched[idx] = 0;

	float3 point = { coordinates[3 * idx], coordinates[3 * idx + 1], coordinates[3 * idx + 2] };

	if (!in_frustum(idx, point))
		return;

	float my_radius = radius / point.z;
	uint2 rect_min, rect_max;
	float2 point_image = { ndc2Pix(point.x, W), ndc2Pix(point.y, H) };
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	depths[idx] = point.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

template <uint32_t CHANNELS> 
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	//input
	const uint2* __restrict__ ranges, 
	const uint32_t* __restrict__ point_list, 
	int W, int H, 
	const float2* __restrict__ points_xy_image, 
	const float* __restrict__ features, 
	const int* __restrict__ radii,
	//output
	float* __restrict__ final_T, 
	uint32_t* __restrict__ n_contrib, 
	float* __restrict__ out_color) 
{
	int Channels = CHANNELS + 1;
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };


	bool inside = pix.x < W&& pix.y < H;

	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS + 1] = { 0 };

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;

			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float d2 = d.x * d.x + d.y * d.y;

			float alpha = min(0.99f, features[collected_id[j] * Channels + 3] * (1 - d2 / (radii[collected_id[j]] * radii[collected_id[j]])));

			if (alpha < 1.0f / 255.0f) //太透明则忽略
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			for (int ch = 0; ch < 3; ch++)
				C[ch] += features[collected_id[j] * Channels + ch] * alpha * T;

			T = test_T;

			last_contributor = contributor;
		}
	}

	float bg_color[3] = {0, 0, 0};
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < 3; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	//input
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* points_xy_image,
	const float* features,
	const int* radii,
	//output
	float* final_T,
	uint32_t* n_contrib,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		//input
		ranges,
		point_list,
		W, H,
		points_xy_image,
		features,
		radii,
		//output
		final_T,
		n_contrib,
		out_color);
}

void FORWARD::preprocess(
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
	uint32_t* tiles_touched)
{
	preprocessCUDA<NUM_CHANNELS> << <(N + 255) / 256, 256 >> > (
		//input
		N,
		points,
		W, H,
		radius,
		grid,
		//output
		radii,
		depths,
		points_xy_image,
		tiles_touched
		);
}