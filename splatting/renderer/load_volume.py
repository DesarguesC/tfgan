import numpy as np

def read_raw_volume(file_path, dimensions, dtype=np.uint8):
    """
    从给定的.raw文件中读取三维体数据。

    参数：
    - file_path (str)：raw 文件的路径
    - dimensions (tuple)：体数据的维度，例如 (x_dim, y_dim, z_dim)
    - dtype：数据类型，默认为 np.float32

    返回：
    - volume (np.ndarray)：三维 NumPy 数组，包含体数据
    """
    volume_size = np.prod(dimensions)
    
    # 读取 raw 文件
    with open(file_path, 'rb') as f:
        volume_data = np.frombuffer(f.read(), dtype=dtype)
    
    if volume_data.size != volume_size:
        raise ValueError("Specified dimensions do not match the file size.")
    
    # 将一维数组重塑为三维数组
    volume = np.reshape(volume_data, dimensions)
    
    return volume

def volume_to_points(volume):
    """将体数据转换为颜色点集合"""
    
    dims = volume.shape
    points = []
    
    # 遍历每个体素
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                
                # 获取体素的密度值
                density = volume[x, y, z]
                if(density > 180):
                # 添加颜色点（x, y, z, density
                    points.append([x/dims[0]-0.5, y/dims[1]-0.5, z/dims[2]-0.5, 1, density])
    
    return np.array(points)

# 使用示例
if __name__ == "__main__":
    file_path = r"D:/files/study/splatting_pytorch/nucleon_41x41x41_uint8.raw"
    dimensions = (41, 41, 41)  # 体数据维度
    volume = read_raw_volume(file_path, dimensions)
    points = volume_to_points(volume)
    print("Volume shape:", volume.shape)
    print("First voxel value:", volume[21, 21, 21])
