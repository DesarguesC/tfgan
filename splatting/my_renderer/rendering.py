from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple, Union

from .cameras import try_get_projection_transform
from .rendering_voxels import rendering

import torch
import torch.nn as nn
   
@dataclass
class RenderingSettings:
    image_width: int = 512
    image_height: int = 512
    radius: float = 100

class Rendering(nn.Module):
    def __init__(self, cameras=None, rendering_settings=None) -> None:

        super().__init__()
        if rendering_settings is None:
            rendering_settings = RenderingSettings()

        self.cameras = cameras
        self.rendering_settings = rendering_settings

    def transform(self, voxels, **kwargs):
        cameras = self.cameras
        
        voxs_view = cameras.get_world_to_view_transform(**kwargs).transform_points(voxels)
        to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = try_get_projection_transform(cameras, kwargs)
        if projection_transform is not None:
            projection_transform = projection_transform.compose(to_ndc_transform)
            voxs_ndc = projection_transform.transform_points(voxs_view)
        else:
            voxs_proj = cameras.transform_points(voxels)
            voxs_ndc = to_ndc_transform.transform_points(voxs_proj)
        voxs_ndc[..., 2] = voxs_view[..., 2]
        return voxs_ndc

    def to(self, device):
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def forward(self, voxels, **kwargs):
        ndc_xy_view_zs = self.transform(voxels[:, :3], **kwargs)
        rendering_settings = self.rendering_settings
        image = rendering(
            ndc_xy_view_zs,
            voxels[:, 3:],
            image_width = rendering_settings.image_width,
            image_height = rendering_settings.image_height,
            radius = rendering_settings.radius,
        )
        return image