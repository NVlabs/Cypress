##
# @file   setup.py
# @author Yibo Lin
# @date   Jun 2018
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os 
import sys

cuda_arch = '${CMAKE_CUDA_FLAGS}'
print("cuda_arch = %s" % (cuda_arch))

def add_prefix(filename):
    return os.path.join('${CMAKE_CURRENT_SOURCE_DIR}/src', filename)

modules = []

modules.extend([
    CppExtension('electric_potential_cpp', 
        [
            add_prefix('electric_density_map.cpp'), 
            add_prefix('electric_force.cpp')
            ]),
    ])

if not "${CUDA_FOUND}" or "${CUDA_FOUND}".upper() == 'TRUE': 
    modules.extend([
            CUDAExtension('electric_potential_cuda', 
                [
                    add_prefix('electric_density_map_cuda.cpp'),
                    add_prefix('electric_density_map_cuda_kernel.cu'),
                    add_prefix('electric_force_cuda.cpp'), 
                    add_prefix('electric_force_cuda_kernel.cu'),
                    ], 
                libraries=['cusparse', 'culibos'],
                extra_compile_args={
                    'cxx': ['-O2'], 
                    'nvcc': [cuda_arch]
                    }
                ),
        ])

setup(
        name='electric_potential',
        ext_modules=modules,
        cmdclass={
            'build_ext': BuildExtension
            })
