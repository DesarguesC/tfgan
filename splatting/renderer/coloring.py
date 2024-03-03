import torch
import numpy as np
from . import load_volume
import torch.nn as nn

class FixedGaussianTFColoring(nn.Module):
    def __init__(self, mean, std_dev, colors):
        super(FixedGaussianTFColoring, self).__init__()
        self.mean = mean
        self.std_dev = std_dev
        self.colors = colors
        

    def forward(self, points):
        coords = points[:, :3]  # Shape [N, 3]
        x_values = points[:, 3]  # Shape [N]
        
        # Expand dimensions for broadcasting
        x_values = x_values.unsqueeze(-1)  # Shape [N, 1]
        gaussians = torch.exp(-0.5 * ((x_values - self.means) / self.std_devs) ** 2)  # Shape [N, num_peaks]
        
        # Multiply by the colors
        weighted_colors = gaussians.unsqueeze(-1) * self.colors  # Shape [N, num_peaks, 4]
        
        # Sum over the peaks dimension
        final_colors = weighted_colors.sum(dim=1)  # Shape [N, 4]
        
        # Concatenate the coordinates and the final colors
        new_points = torch.cat([coords, final_colors], dim=1)  # Shape [N, 7]
        
        return new_points



class GaussianTFColoring(nn.Module):
    def __init__(self, initial_means, initial_std_devs, initial_colors):
        super(GaussianTFColoring, self).__init__()
        self.means = nn.Parameter(torch.tensor(initial_means))
        self.std_devs = nn.Parameter(torch.tensor(initial_std_devs))
        self.colors = nn.Parameter(torch.tensor(initial_colors))

    def forward(self, points):
        coords = points[:, :4]  # Shape [N, 4]
        x_values = points[:, 4]  # Shape [N]
        
        # Expand dimensions for broadcasting
        x_values = x_values.unsqueeze(-1)  # Shape [N, 1]
        gaussians = torch.exp(-0.5 * ((x_values - self.means) / self.std_devs) ** 2)  # Shape [N, num_peaks]
        
        # Multiply by the colors
        weighted_colors = gaussians.unsqueeze(-1) * self.colors  # Shape [N, num_peaks, 4]
        
        # Sum over the peaks dimension
        final_colors = weighted_colors.sum(dim=1)  # Shape [N, 4]
        
        # Concatenate the coordinates and the final colors
        new_points = torch.cat([coords, final_colors], dim=1)  # Shape [N, 8]
        
        return new_points

# 使用示例
if __name__ == "__main__":
    # 检查 CUDA 是否可用，并设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    file_path = r"D:/files/study/splatting_pytorch/nucleon_41x41x41_uint8.raw"
    dimensions = (41, 41, 41)  # 体数据维度
    volume = load_volume.read_raw_volume(file_path, dimensions)
    points = torch.from_numpy(load_volume.volume_to_points(volume)).to(device)
    

    
    # 使用
    initial_means = [50.0, 150.0]
    initial_std_devs = [10.0, 30.0]
    initial_colors = [[0.5, 0.5, 0.5, 0.5], [0.2, 0.3, 0.1, 0.2]]
    
    gaussian_tf = GaussianTFColoring(initial_means, initial_std_devs, initial_colors).to(device)

    colorpoints = gaussian_tf(points)
    print(colorpoints)




