#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	//input
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, float radius,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ depths,
	const float* __restrict__ features,
	const int* __restrict__  radii,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	//output
	float3* __restrict__ dL_dcoordinates,
	float* __restrict__ dL_dfeatures)
{
	int C4 = C + 1;
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float collected_features[(C + 1) * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float dscreenx_dx = 0.5 * W;
	const float dscreeny_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_depths[block.thread_rank()] = depths[coll_id];
			for (int i = 0; i < C4; i++)
				collected_features[i * BLOCK_SIZE + block.thread_rank()] = features[coll_id * C4 + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float d2 = (d.x*d.x + d.y*d.y);
			const float radii2 = radii[collected_id[j]] * radii[collected_id[j]];

			const float alpha = min(0.99f, features[collected_id[j] * C4 + 3] * (1 - d2 / radii2));
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients 
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_features[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch]; //从后向前混合
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;

				atomicAdd(&(dL_dfeatures[global_id * C4 + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Helpful reusable temporary variables
			const float dalpha_dd2 = -1 / radii2 * features[collected_id[j] * C4 + 3];
			const float dalpha_dradii = 2 * d2 / (radii2 * radii[collected_id[j]]);


			// Update gradients w.r.t. ndc coordinates
			atomicAdd(&dL_dcoordinates[global_id].x, dL_dalpha * dalpha_dd2 * 2 * (xy.x - pixf.x) * dscreenx_dx);
			atomicAdd(&dL_dcoordinates[global_id].y, dL_dalpha * dalpha_dd2 * 2 * (xy.y - pixf.y) * dscreeny_dy);
			atomicAdd(&dL_dcoordinates[global_id].z, dL_dalpha * dalpha_dradii * -radius / collected_depths[j] / collected_depths[j]);

			// Update gradients w.r.t. alpha channel
			atomicAdd(&dL_dfeatures[global_id * C4 + 3], dL_dalpha * (1 - d2 / radii2));
		}
	}
}

void BACKWARD::render(
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
	float* dL_dfeatures)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H, radius,
		points_xy_image,
		depths,
		features,
		radii,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dcoordinates,
		dL_dfeatures
		);
}