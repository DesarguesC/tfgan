/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include "utils/pytorch3d_cutils.h"

// ****************************************************************************
// *                          NAIVE RASTERIZATION                             *
// ****************************************************************************


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizePointsNaiveCuda(
    const torch::Tensor& points,
    const std::tuple<int, int> image_size,
    const torch::Tensor& radius,
    const int points_per_pixel);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePointsNaive(
    const torch::Tensor& points,
    const std::tuple<int, int> image_size,
    const torch::Tensor& radius,
    const int points_per_pixel) {
    CHECK_CUDA(points);
    CHECK_CUDA(radius);
    return RasterizePointsNaiveCuda(
        points,
        image_size,
        radius,
        points_per_pixel);
}

// ****************************************************************************
// *                            BACKWARD PASS                                 *
// ****************************************************************************
torch::Tensor RasterizePointsBackwardCuda(
    const torch::Tensor& points,
    const torch::Tensor& idxs,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists);

torch::Tensor RasterizePointsBackward(
    const torch::Tensor& points,
    const torch::Tensor& idxs,
    const torch::Tensor& grad_zbuf,
    const torch::Tensor& grad_dists) {
    CHECK_CUDA(points);
    CHECK_CUDA(idxs);
    CHECK_CUDA(grad_zbuf);
    CHECK_CUDA(grad_dists);
    return RasterizePointsBackwardCuda(points, idxs, grad_zbuf, grad_dists);
}

// ****************************************************************************
// *                         MAIN ENTRY POINT                                 *
// ****************************************************************************
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizePoints(
    const torch::Tensor& points,
    const std::tuple<int, int> image_size,
    const torch::Tensor& radius,
    const int points_per_pixel,
    const int bin_size,
    const int max_points_per_bin) {
    return RasterizePointsNaive(
        points,
        image_size,
        radius,
        points_per_pixel);
}
