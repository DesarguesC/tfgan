/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <math.h>
#include <cstdio>
#include <sstream>
#include <tuple>
#include "rasterization_utils.cuh"

namespace {
// A little structure for holding details about a pixel.
struct Pix {
  float z; // Depth of the reference point.
  int32_t idx; // Index of the reference point.
  float dist2; // Euclidean distance square to the reference point.
};

__device__ inline bool operator<(const Pix& a, const Pix& b) {
  return a.z < b.z;
}

template <typename PointQ>
__device__ void CheckPixelInsidePoint(
    const float* points, // (P, 3)
    const int p_idx,
    int& q_size,
    float& q_max_z,
    int& q_max_idx,
    PointQ& q,
    const float* radius,
    const float xf,
    const float yf,
    const int K) {
  const float px = points[p_idx * 3 + 0];
  const float py = points[p_idx * 3 + 1];
  const float pz = points[p_idx * 3 + 2];
  const float p_radius = radius[p_idx];
  const float radius2 = p_radius * p_radius;
  if (pz < 0)
    return; // Don't render points behind the camera
  const float dx = xf - px;
  const float dy = yf - py;
  const float dist2 = dx * dx + dy * dy;
  if (dist2 < radius2) {
    if (q_size < K) {
      // Just insert it
      q[q_size] = {pz, p_idx, dist2};
      if (pz > q_max_z) {
        q_max_z = pz;
        q_max_idx = q_size;
      }
      q_size++;
    } else if (pz < q_max_z) {
      // Overwrite the old max, and find the new max
      q[q_max_idx] = {pz, p_idx, dist2};
      q_max_z = pz;
      for (int i = 0; i < K; i++) {
        if (q[i].z > q_max_z) {
          q_max_z = q[i].z;
          q_max_idx = i;
        }
      }
    }
  }
}
} // namespace
// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************

__global__ void RasterizePointsNaiveCudaKernel(
    const float* points, // (P, 3)
    const float* radius,
    const int P,
    const int N,
    const int H,
    const int W,
    const int K,
    int32_t* point_idxs, // (N, H, W, K)
    float* zbuf, // (N, H, W, K)
    float* pix_dists) { // (N, H, W, K)
  // Simple version: One thread per output pixel
  const int num_threads = gridDim.x * blockDim.x;
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = tid; i < N * H * W; i += num_threads) {
    // Convert linear index to 3D index
    const int pix_idx = i % (H * W);

    // Reverse ordering of the X and Y axis as the camera coordinates
    // assume that +Y is pointing up and +X is pointing left.
    const int yi = H - 1 - pix_idx / W;
    const int xi = W - 1 - pix_idx % W;

    // screen coordinates to ndc coordinates of pixel.
    const float xf = PixToNonSquareNdc(xi, W, H);
    const float yf = PixToNonSquareNdc(yi, H, W);

    Pix q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;

    // Using the batch index of the thread get the start and stop
    // indices for the points.
    const int64_t point_start_idx = 0;
    const int64_t point_stop_idx = P;

    for (int p_idx = point_start_idx; p_idx < point_stop_idx; ++p_idx) {
      CheckPixelInsidePoint(
          points, p_idx, q_size, q_max_z, q_max_idx, q, radius, xf, yf, K);
    }
    BubbleSort(q, q_size);
    int idx = pix_idx * K;
    for (int k = 0; k < q_size; ++k) {
      point_idxs[idx + k] = q[k].idx;
      zbuf[idx + k] = q[k].z;
      pix_dists[idx + k] = q[k].dist2;
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> RasterizePointsNaiveCuda(
    const at::Tensor& points, // (P. 3)
    const std::tuple<int, int> image_size,
    const at::Tensor& radius,
    const int points_per_pixel) {

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(
      points.ndimension() == 2 && points.size(1) == 3,
      "points must have dimensions (num_points, 3)");

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);
  const int K = points_per_pixel;

  if (K > kMaxPointsPerPixel) {
    std::stringstream ss;
    ss << "Must have points_per_pixel <= " << kMaxPointsPerPixel;
    AT_ERROR(ss.str());
  }

  auto int_opts = points.options().dtype(at::kInt);
  auto float_opts = points.options().dtype(at::kFloat);
  at::Tensor point_idxs = at::full({1, H, W, K}, -1, int_opts);
  at::Tensor zbuf = at::full({1, H, W, K}, -1, float_opts);
  at::Tensor pix_dists = at::full({1, H, W, K}, -1, float_opts);


  if (point_idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(point_idxs, zbuf, pix_dists);
  }

  const size_t blocks = 1024;
  const size_t threads = 64;
  const int P = points.size(0);

  RasterizePointsNaiveCudaKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      radius.contiguous().data_ptr<float>(),
      P,
      1,
      H,
      W,
      K,
      point_idxs.contiguous().data_ptr<int32_t>(),
      zbuf.contiguous().data_ptr<float>(),
      pix_dists.contiguous().data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(point_idxs, zbuf, pix_dists);
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
// TODO(T55115174) Add more documentation for backward kernel.
__global__ void RasterizePointsBackwardCudaKernel(
    const float* points, // (P, 3)
    const int32_t* idxs, // (N, H, W, K)
    const int N,
    const int P,
    const int H,
    const int W,
    const int K,
    const float* grad_zbuf, // (N, H, W, K)
    const float* grad_dists, // (N, H, W, K)
    float* grad_points) { // (P, 3)
  // Parallelized over each of K points per pixel, for each pixel in images of
  // size H * W, for each image in the batch of size N.
  int num_threads = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < N * H * W * K; i += num_threads) {
    // const int n = i / (H * W * K); // batch index (not needed).
    const int yxk = i % (H * W * K);
    const int yi = yxk / (W * K);
    const int xk = yxk % (W * K);
    const int xi = xk / K;
    // k = xk % K (We don't actually need k, but this would be it.)
    // Reverse ordering of X and Y axes.
    const int yidx = H - 1 - yi;
    const int xidx = W - 1 - xi;

    const float xf = PixToNonSquareNdc(xidx, W, H);
    const float yf = PixToNonSquareNdc(yidx, H, W);

    const int p = idxs[i];
    if (p < 0)
      continue;
    const float grad_dist2 = grad_dists[i];
    const int p_ind = p * 3; // index into packed points tensor
    const float px = points[p_ind + 0];
    const float py = points[p_ind + 1];
    const float dx = px - xf;
    const float dy = py - yf;
    const float grad_px = 2.0f * grad_dist2 * dx;
    const float grad_py = 2.0f * grad_dist2 * dy;
    const float grad_pz = grad_zbuf[i];
    atomicAdd(grad_points + p_ind + 0, grad_px);
    atomicAdd(grad_points + p_ind + 1, grad_py);
    atomicAdd(grad_points + p_ind + 2, grad_pz);
  }
}

at::Tensor RasterizePointsBackwardCuda(
    const at::Tensor& points, // (N, P, 3)
    const at::Tensor& idxs, // (N, H, W, K)
    const at::Tensor& grad_zbuf, // (N, H, W, K)
    const at::Tensor& grad_dists) { // (N, H, W, K)

  // Check inputs are on the same device
  at::TensorArg points_t{points, "points", 1}, idxs_t{idxs, "idxs", 2},
      grad_zbuf_t{grad_zbuf, "grad_zbuf", 3},
      grad_dists_t{grad_dists, "grad_dists", 4};
  at::CheckedFrom c = "RasterizePointsBackwardCuda";
  at::checkAllSameGPU(c, {points_t, idxs_t, grad_zbuf_t, grad_dists_t});
  at::checkAllSameType(c, {points_t, grad_zbuf_t, grad_dists_t});
  // This is nondeterministic because atomicAdd
  at::globalContext().alertNotDeterministic("RasterizePointsBackwardCuda");
  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(points.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int P = points.size(0);
  const int N = idxs.size(0);
  const int H = idxs.size(1);
  const int W = idxs.size(2);
  const int K = idxs.size(3);

  at::Tensor grad_points = at::zeros({P, 3}, points.options());

  if (grad_points.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_points;
  }

  const size_t blocks = 1024;
  const size_t threads = 64;

  RasterizePointsBackwardCudaKernel<<<blocks, threads, 0, stream>>>(
      points.contiguous().data_ptr<float>(),
      idxs.contiguous().data_ptr<int32_t>(),
      N,
      P,
      H,
      W,
      K,
      grad_zbuf.contiguous().data_ptr<float>(),
      grad_dists.contiguous().data_ptr<float>(),
      grad_points.contiguous().data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  return grad_points;
}
