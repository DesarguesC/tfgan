from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import my_cuda_extension

from ..utils import parse_image_size

def rasterize_points_pro( 
    xyz_points,
    image_size: Union[int, List[int], Tuple[int, int]] = 512,
    radius: Union[float, List, Tuple, torch.Tensor] = 0.01,
    block_size: int = 16,
):
    radius = _format_radius(radius, xyz_points)

    im_size = parse_image_size(image_size)
    blk_size = (block_size, block_size)

    return _RasterizePointsPro.apply(
        xyz_points,
        im_size,
        radius,
        blk_size,
    )


def _format_radius(
    radius: Union[float, List, Tuple, torch.Tensor], xyz_points
) -> torch.Tensor:

    radius = torch.full((xyz_points.shape[0],), fill_value=radius).type_as(xyz_points)
    return radius


class _RasterizePointsPro(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points,
        image_size: Union[List[int], Tuple[int, int]] = (256, 256),
        radius: Union[float, torch.Tensor] = 0.01,
        block_size: Tuple[int, int] = (16, 16),
    ):
        args = (
            points,
            image_size,
            radius,
            block_size,
        )
        idx, zbuf, zbuf_idx, dists = my_cuda_extension.rasterize_points_pro(*args)
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