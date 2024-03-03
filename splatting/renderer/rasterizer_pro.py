#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Union

from .cameras import try_get_projection_transform

import torch
import torch.nn as nn

from .rasterize_points_pro import rasterize_points_pro

class  PointFragments(NamedTuple):
    idx: torch.Tensor
    dists: torch.Tensor
    


@dataclass
class PointsRasterizationSettingsPro:
    image_size: Union[int, Tuple[int, int]] = 512
    radius: Union[float, torch.Tensor] = 0.01
    block_size: int = 16


class PointsRasterizerPro(nn.Module):
    def __init__(self, cameras=None, raster_settings=None) -> None:

        super().__init__()
        if raster_settings is None:
            raster_settings = PointsRasterizationSettingsPro()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, xyz_points, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of PointsRasterizer"
            raise ValueError(msg)

        eps = kwargs.get("eps", None)
        pts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            xyz_points, eps=eps
        )
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = try_get_projection_transform(cameras, kwargs)
        if projection_transform is not None:
            projection_transform = projection_transform.compose(to_ndc_transform)
            pts_ndc = projection_transform.transform_points(pts_view, eps=eps)
        else:
            pts_proj = cameras.transform_points(xyz_points, eps=eps)
            pts_ndc = to_ndc_transform.transform_points(pts_proj, eps=eps)

        pts_ndc[..., 2] = pts_view[..., 2]
        return pts_ndc

    def to(self, device):
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def forward(self, xyz_points, **kwargs) -> PointFragments:
        points_proj = self.transform(xyz_points, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        idx, zbuf, zbuf_idx, dists2 = rasterize_points_pro(
            points_proj,
            image_size=raster_settings.image_size,
            radius=raster_settings.radius,
            block_size=raster_settings.block_size,
        )
        return PointFragments(idx=idx, zbuf=zbuf, zbuf_idx=zbuf_idx, dists=dists2)
