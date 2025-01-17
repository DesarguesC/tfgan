#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


# A renderer class should be initialized with a
# function for rasterization and a function for compositing.
# The rasterizer should:
#     - transform inputs from world -> screen space
#     - rasterize inputs
#     - return fragments
# The compositor can take fragments as input along with any other properties of
# the scene and generate images.

# E.g. rasterize inputs and then shade
#
# fragments = self.rasterize(point_clouds)
# images = self.compositor(fragments, point_clouds)
# return images


class PointsRenderer(nn.Module): #光栅化是目前需要重构的重要部分，ta的输入需要替换成颜色映射后的体数据
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, points, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(points[:,:3], **kwargs) #光柵化 片段包含每个像素关联点的三个信息（idx zbuf dist）

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),      # N K H W
            weights,
            points[:,4:8], # (num_points, C)
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images
