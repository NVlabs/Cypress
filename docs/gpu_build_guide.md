# Cypress GPU Build Documentation

This document outlines the steps required to rebuild the Cypress PCB placement tool with GPU acceleration enabled. Cypress leverages NVIDIA GPUs via PyTorch and custom CUDA kernels for high-performance placement.

## Prerequisites

Before starting the build process, ensure your system meets the following requirements:

### 1. Hardware
*   **NVIDIA GPU**: A GPU with Pascal architecture (Compute Capability 6.0) or newer is recommended (e.g., GTX 10-series, RTX-series, Tesla P/V/T/A-series).

### 2. Software
*   **NVIDIA Driver**: Latest stable drivers compatible with your CUDA version.
*   **CUDA Toolkit**: Version 9.0 to 11.x (Version 11.0+ is recommended for newer GPUs).
*   **cuDNN**: Compatible with your CUDA version.
*   **C++ Compiler**: GCC/G++ 7.0 or newer (must be compatible with the installed CUDA version).
*   **CMake**: Version 3.14 or newer.
*   **Python**: Version 3.6 or newer.

## Step-by-Step Rebuild Guide

### 1. Set Up the Environment

Ensure `nvcc` and CUDA libraries are in your environment paths. Add the following to your `~/.bashrc` or execute them in your terminal:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Verify the installation:
```bash
nvcc --version
```

### 2. Install PyTorch with CUDA Support

It is critical that the PyTorch version installed has CUDA enabled. Use `uv` or `pip` to install the appropriate version:

```bash
# Example for CUDA 11.8
uv run pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify that PyTorch can see the GPU:
```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
# Should output: True
```

### 3. Install Other Dependencies

Install the remaining Python requirements:
```bash
uv run pip install -r requirements.txt
```

### 4. Build C++ Extensions

Cypress uses CMake to manage its custom CUDA kernels and C++ extensions.

1.  **Clear previous builds** (optional but recommended):
    ```bash
    rm -rf build
    mkdir build
    cd build
    ```

2.  **Configure with CMake**:
    The build system automatically detects if CUDA is available via PyTorch and the system `nvcc`.
    ```bash
    cmake .. -DCMAKE_INSTALL_PREFIX=../
    ```
    *Note: During this step, watch for the message `TORCH_ENABLE_CUDA=1`. If it shows `0`, the GPU build will not proceed.*

3.  **Compile and Install**:
    ```bash
    make -j$(nproc)
    make install
    ```
    This will compile the CUDA kernels and place the resulting `.so` files into the `dreamplace/ops` directory.

## Findings & Recommendations

### Common Build Issues

1.  **ABI Mismatch**:
    If you encounter linking errors related to `std::__cxx11`, you may need to toggle the `_GLIBCXX_USE_CXX11_ABI` flag in `CMakeLists.txt`:
    ```cmake
    set(CMAKE_CXX_ABI 0) # or 1
    ```

2.  **Missing `libtorch_cuda.so`**:
    This occurs when the PyTorch version installed is CPU-only. Always verify with `torch.cuda.is_available()` before building.

3.  **CUDA Architecture Mismatch**:
    The `CMakeLists.txt` is configured with a default list of architectures: `6.0 6.1 7.0 7.5 8.0`. If you have a newer GPU (e.g., RTX 30/40 series), you may need to add `8.6` or `8.9` to `CUDA_ARCH_LIST` in the root `CMakeLists.txt`.

### Running with GPU

Once built, enable GPU execution in your JSON configuration file:

```json
{
    "gpu": 1,
    "gpu_id": 0
}
```

If multiple GPUs are available, you can specify which one to use via `gpu_id`.

## Verification of GPU Usage

During execution, the logs should indicate if GPU is being used. You can also monitor GPU utilization using `nvidia-smi`:

```bash
watch -n 1 nvidia-smi
```
