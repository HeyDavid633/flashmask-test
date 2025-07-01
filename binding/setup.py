# 2025.6.24 
# 自动化的setup.py文件 尝试绑定 FlashAttention为我自己的算子 
#
# python setup.py install
# 如果不在当前目录产生so文件：
# python setup.py build_ext --inplace
# 新增算子不需再改动这个文件，通过get_extension()自动提取源文件名称，并注册为新算子
# 注意新增的算子 名称前缀不要太相似(有太多公共前缀)
# 
# 对于不同的功能，可能需要重命名 setup(name = 'xxx'
# 编译标志中 sm 的型号可能需要依据平台而改变  A100:sm80 | 4080:sm89 | 3090:sm86
import os
import sys
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if len(sys.argv) > 1:
    cuda_arch = sys.argv.pop(1)  # 取出第一个参数
    try:
        int(cuda_arch)  # 确保输入的是整数
    except ValueError:
        print("Error: CUDA architecture must be a number (e.g., 80, 89).")
        sys.exit(1)
else:
    cuda_arch = "80"  # 默认架构

print(f" ... ... Compiling for CUDA architecture: sm_{cuda_arch}")

extra_compile_args = {
    "nvcc" : ["-O3", "-w",
              "-I/usr/local/cuda/include", 
              f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}"]  #此处gencode后面的等号不可以省
}
extra_link_args = []

def get_extensions():
    
    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, "ops", "src")
    
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*_cuda.cu")))
    
    # 从文件名中提取算子名称
    extension_names = set()
    for source in sources + cuda_sources:
        base_name = os.path.basename(source)
        name, _ = os.path.splitext(base_name)
        if name.endswith("_cuda"):
            name = name[:-5]  # 去掉 "_cuda" 后缀
        extension_names.add(name)
            
    ext_modules = []
    for name in extension_names:
        # 动态生成每个扩展的源文件列表
        extension_sources = [s for s in sources + cuda_sources if name in os.path.basename(s)]
        ext_modules.append(
            CUDAExtension(
                name=name,
                sources=extension_sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args
            )
        )
    
    return ext_modules

setup(
    name='PPoPP25-Attention-Kernel-Binding',
    packages=find_packages(),
    version='0.1.2',
    author='PPoPP25 Attention-Kernel Bindings by David',
    
    ext_modules=get_extensions(),
    
    install_requires=["torch"],
    description="customed operator linked with cutlass implement",
    
    cmdclass={'build_ext': BuildExtension}
)