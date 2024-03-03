import torch
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.models as models
from torch import nn
import splatting.my_renderer.loading_volume as vol
import splatting.my_renderer.cameras as camera
import splatting.my_renderer.mapping as mapping
import splatting.my_renderer.rendering as rendering
import splatting.my_renderer.my_renderer as my_renderer

def local_creator(resolution, colors, device):
    # alpha_ = 0.09 * torch.randn(1,1).to(device) # 方差越小越好, 但0.又不行, 0.1可(基本上最佳), 0.05-0.09不可
    # 初值对结果影响蛮大的
    alpha_ = torch.tensor(0.5).to(device).reshape(1,1)

    t = 0.1 * torch.randn(1,1).to(device)
    while torch.sum(t) < 0:
        t = 0.1 * torch.randn(1,1).to(device)
    alpha_ = t

    alpha_ = nn.Parameter(alpha_.clone().detach().requires_grad_(True))
    return mapping.TextureTFMapping(resolution, colors, device=device).to(device), alpha_


class TFMapping(nn.Module):
    """
        传输函数控制类
        self.GlobalTF: 全局传输函数
        self.tag_list: 已经归类的tag, 0为初始状态, 对应GlobalTF, 其余tag通过用户选择后传入
        self.LocalTF: 已构建的局部传输函数, self.LocalTF[0]为GlobalTF
        self.alpha: 存储
        反向传播时, 认为只实现了
    """
    def __init__(self, resolution, colors, device, lock=False):  
        super(TFMapping, self).__init__()  
        self.device = device
        self.resolution = resolution
        # print(self.colors.requires_grad)
        self.GlobalTF = mapping.TextureTFMapping(resolution, colors, device=device, lock=lock).to(device)
        self.colors = self.GlobalTF.colors
        print(f'cloned -> {self.colors is self.GlobalTF.colors}')
        self.offsets = torch.linspace(-1, 254, 256).to(self.device)  # Initialize offsets evenly  
        self.tag_list = [0]
        self.LocalTF = [self.GlobalTF]
        self.print_sensitivity = False
        self.do_print()
    
    def do_print(self):
        self.print_sensitivity = True
    def undo_print(self):
        self.print_sensitivity = False
    
    def do_print(self):
        self.print_sensitivity = True
    def undo_print(self):
        self.print_sensitivity = False
    
    def __str__(self):
        return f'LocalTF num: {len(self.LocalTF)-1}'

    @torch.no_grad()
    def flash(self, points):
        """
            刷新当前控制类
            根据points中新增的tag (仅一), 创建LocalTF局部传输函数 以及 对应的控制权重
        """
        # e.g. points: torch [0,0,0,0,1,0,0,0,2,0,3] ...
        target = torch.max(points[:, 4].long()).item()
        assert len(self.LocalTF) == target, f'target tag: {target}, len(localTF) = {len(self.LocalTF)}'
        local_, alpha_ = local_creator(self.resolution, self.colors, self.device)
        self.LocalTF.append(local_)
        setattr(self, f'alpha_{target}', alpha_) # 方便adam加入计算图追踪梯度
        setattr(self, f'local_color_{target}', local_.colors) # 方便adam加入计算图追踪梯度
        assert len(self.LocalTF) == target + 1, f'LocalTF ?'
        self.tag_list.append(target)

    def forward(self, points):  
        """
            反向传播
            需保证每次优化开始前, 使用控制类对体素points进行flash
        """
        coords = points[:, :3]          # Shape [N, 3], normalized  
        densitys = points[:, 3].long()  # Shape [N], converted to long for indexing  
        tags = torch.max(points[:, 4].long()).item()      # Shape [N], tagged for mask
        # Calculate the index into the lookup table  
        color_index = (densitys * (self.resolution - 1) // 255).clamp(0, self.resolution - 1).to(self.device)  # Shape [N]  -> global
        # print(f'color_index.shape = {color_index.shape}') # Shape [N] (long: 0 ~ 63)
        # print(f'color_index = {color_index}')
        colors = self.colors[color_index].clamp(0., 1.)  # Clamp colors between 0.0 and 1.0  -> globalTF color

        # Concatenate the coordinates and the final colors  
        colored_voxels = torch.cat([coords, colors], dim=1).clone().detach().requires_grad_(True)  # Shape [N, 7]  -> global

        assert tags in self.tag_list
        local_colored_voxels = self.LocalTF[tags](points) # .colors[color_index].clamp(0., 1.)
        alpha_i = getattr(self, f'alpha_{tags}')
        colored_voxels = alpha_i * colored_voxels + (1. - alpha_i) * local_colored_voxels

        return colored_voxels if not self.print_sensitivity else (colored_voxels, color_index)


