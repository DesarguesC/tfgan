from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 假设你的CUDA算子的源文件和头文件如下：
cpp_files = [
    "D:/files/study/splatting_pytorch/splatting/Ext/csrc/ext.cpp"
]
cu_files = [
    "D:/files/study/splatting_pytorch/splatting/Ext/csrc/rasterize_points/rasterize_points.cu",
    "D:/files/study/splatting_pytorch/splatting/Ext/csrc/compositing/alpha_composite.cu"
]

setup(
    name="my_cuda_extension",
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="my_cuda_extension",
            sources=cpp_files + cu_files,
            extra_compile_args={"cxx": [], 
                                "nvcc": ["-DCUDA_HAS_FP16=1",
                                        "-D__CUDA_NO_HALF_OPERATORS__",
                                        "-D__CUDA_NO_HALF_CONVERSIONS__",
                                        "-D__CUDA_NO_HALF2_OPERATORS__",]}, 
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
