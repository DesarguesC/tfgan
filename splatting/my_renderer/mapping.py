import torch
import numpy as np
from . import loading_volume
import torch.nn as nn

class GaussianTFMapping(nn.Module):
    def __init__(self, initial_means, initial_std_devs, initial_colors):
        super(GaussianTFMapping, self).__init__()
        self.means = nn.Parameter(torch.tensor(initial_means))
        self.std_devs = nn.Parameter(torch.tensor(initial_std_devs))
        self.colors = nn.Parameter(torch.tensor(initial_colors))

    def forward(self, points):
        coords = points[:, :3]  # Shape [N, 3]
        densitys = points[:, 3]  # Shape [N]
        
        # Expand dimensions for broadcasting
        densitys = densitys.unsqueeze(-1)  # Shape [N, 1]
        gaussians = torch.exp(-0.5 * ((densitys - self.means) / self.std_devs) ** 2)  # Shape [N, num_peaks]
        
        # Multiply by the colors
        weighted_colors = gaussians.unsqueeze(-1) * self.colors  # Shape [N, num_peaks, 4]
        
        # Sum over the peaks dimension
        final_colors = weighted_colors.sum(dim=1)  # Shape [N, 4]
        
        final_colors = final_colors.clamp(0.0, 1.0)  # Clamp colors between 0.0 and 1.0  
        
        # Concatenate the coordinates and the final colors
        colored_voxels = torch.cat([coords, final_colors], dim=1)  # Shape [N, 7]
        
        return colored_voxels
    
class TextureTFMapping(nn.Module):  
    def __init__(self, resolution, colors, device, lock=False):  
        super(TextureTFMapping, self).__init__()  
        self.device = device
        self.resolution = resolution  
        self.colors = colors if lock else nn.Parameter(colors.clone().detach().requires_grad_(True))
        self.offsets = torch.linspace(-1, 254, 256).to(self.device)  # Initialize offsets evenly  
  
    def forward(self, points):  
        coords = points[:, :3]  # Shape [N, 3]  
        densitys = points[:, 3].long()  # Shape [N], converted to long for indexing  
  
        # Calculate the index into the lookup table  
        color_index = (densitys * (self.resolution - 1) // 255).clamp(0, self.resolution - 1)  # Shape [N]  
        color_index = color_index.to(self.device)
        
        # Apply the linear function  
        colors = self.colors[color_index]# * (densitys.unsqueeze(-1) - self.offsets[densitys].unsqueeze(-1))  # Shape [N, 4]  
        
        colors = colors.clamp(0.0, 1.0)  # Clamp colors between 0.0 and 1.0  
  
        # Concatenate the coordinates and the final colors  
        colored_voxels = torch.cat([coords, colors], dim=1)  # Shape [N, 7]  
  
        return colored_voxels  

# test
if __name__ == "__main__":
    # 检查 CUDA 是否可用，并设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    file_path = r"D:/files/study/splatting_pytorch/nucleon_41x41x41_uint8.raw"
    dimensions = (41, 41, 41)  # 体数据维度
    volume = loading_volume.read_raw_volume(file_path, dimensions)
    voxels = torch.from_numpy(loading_volume.volume_to_voxels(volume)).to(device)
    
    # 使用
    initial_means = [50.0, 150.0]
    initial_std_devs = [10.0, 30.0]
    initial_colors = [[0.5, 0.5, 0.5, 0.5], [0.2, 0.3, 0.1, 0.2]]
    
    gaussian_tf = GaussianTFMapping(initial_means, initial_std_devs, initial_colors).to(device)

    colored_voxels = gaussian_tf(voxels)
    print(colored_voxels)




