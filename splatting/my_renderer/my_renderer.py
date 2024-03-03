import torch
import torch.nn as nn

class Renderer(nn.Module): 
    def __init__(self, mapping, rendering) -> None:
        super().__init__()
        self.mapping = mapping
        self.rendering = rendering

    def forward(self, voxels, default=True, **kwargs) -> torch.Tensor:
        if default:
            colored_voxels = self.mapping(voxels, **kwargs).float()
            image = self.rendering(colored_voxels)
            return image 
        else:
            colored_voxels, color_index = self.mapping(voxels, **kwargs)
            colored_voxels, color_index = colored_voxels.float(), color_index.long()
            image = self.rendering(colored_voxels)
            return (image, color_index, colored_voxels)

    
    

