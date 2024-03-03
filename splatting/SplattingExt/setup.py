from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

cpp_files = [
    "./splatting/SplattingExt/csrc/ext.cpp"
]
cu_files = [
    "./splatting/SplattingExt/csrc/splatting/rendering_impl.cu",
    "./splatting/SplattingExt/csrc/splatting/forward.cu",
    "./splatting/SplattingExt/csrc/splatting/backward.cu",
    "./splatting/SplattingExt/csrc/splatting/rendering_function.cu",
]

setup(
    name="cuda_renderer",
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cuda_renderer",
            sources=cpp_files + cu_files,
            extra_compile_args={"cxx": [], 
                                "nvcc": []}, 
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
