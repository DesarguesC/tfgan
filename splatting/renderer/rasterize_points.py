# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import my_cuda_extension

from ..utils import parse_image_size


# Maximum number of faces per bins for
# coarse-to-fine rasterization
kMaxPointsPerBin = 22


def rasterize_points( #上级代码入口
    xyz_points,
    image_size: Union[int, List[int], Tuple[int, int]] = 256,
    radius: Union[float, List, Tuple, torch.Tensor] = 0.01,
    points_per_pixel: int = 8,
    bin_size: Optional[int] = None,
    max_points_per_bin: Optional[int] = None,
):
    radius = _format_radius(radius, xyz_points)

    im_size = parse_image_size(image_size)
    max_image_size = max(*im_size)

    if bin_size is None:
        if not xyz_points.is_cuda:
            # Binned CPU rasterization not fully implemented
            bin_size = 0
        else:
            bin_size = int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))

    if bin_size != 0:
        # There is a limit on the number of points per bin in the cuda kernel.
        points_per_bin = 1 + (max_image_size - 1) // bin_size
        if points_per_bin >= kMaxPointsPerBin:
            raise ValueError(
                "bin_size too small, number of points per bin must be less than %d; got %d"
                % (kMaxPointsPerBin, points_per_bin)
            )

    if max_points_per_bin is None:
        max_points_per_bin = int(max(10000, xyz_points.shape[0] / 5))

    # Function.apply cannot take keyword args, so we handle defaults in this
    # wrapper and call apply with positional args only
    return _RasterizePoints.apply(
        xyz_points,
        im_size,
        radius,
        points_per_pixel,
        bin_size,
        max_points_per_bin,
    )


def _format_radius(
    radius: Union[float, List, Tuple, torch.Tensor], xyz_points
) -> torch.Tensor:

    radius = torch.full((xyz_points.shape[0],), fill_value=radius).type_as(xyz_points)
    return radius


class _RasterizePoints(torch.autograd.Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        points,
        image_size: Union[List[int], Tuple[int, int]] = (256, 256),
        radius: Union[float, torch.Tensor] = 0.01,
        points_per_pixel: int = 8,
        bin_size: int = 0,
        max_points_per_bin: int = 0,
    ):
        # TODO: Add better error handling for when there are more than
        # max_points_per_bin in any bin.
        args = (
            points,
            image_size,
            radius,
            points_per_pixel,
            bin_size,
            max_points_per_bin,
        )
        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        idx, zbuf, dists = my_cuda_extension.rasterize_points(*args)
        ctx.save_for_backward(points, idx)
        ctx.mark_non_differentiable(idx)
        return idx, zbuf, dists

    @staticmethod
    def backward(ctx, grad_idx, grad_zbuf, grad_dists):
        grad_points = None #把颜色属性单独拆出来
        grad_image_size = None
        grad_radius = None
        grad_points_per_pixel = None
        grad_bin_size = None
        grad_max_points_per_bin = None
        points, idx = ctx.saved_tensors
        args = (points, idx, grad_zbuf, grad_dists)
        grad_points = my_cuda_extension.rasterize_points_backward(*args)
        grads = (
            grad_points,
            grad_image_size,
            grad_radius,
            grad_points_per_pixel,
            grad_bin_size,
            grad_max_points_per_bin,
        )
        return grads