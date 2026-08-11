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

# Build 
build doesn't work in local environment. lot of libraries and modules are outdated.
configure and build everything in docker container
cypress framework should be ran inside container
do not setup uv python environment in host machine
Simply share the project repo with the container through docker volume mount, run the framework inside container, collect artefacts from host machine

## Steps to build and run Cypress
- DO NOT attempt to build or run Cypress from host machine
    - Cypress is originally built using old dependencies 
    - It is far easier to build Cypress using docker devcontainer 
    - We may upgrade Cypress dependencies and build it in native system after stabilizing the pipeline
    - For now, best solution is to use the container for build and run
- [docker compose](../docker-compose.yaml) file available in the repo is configured to mount the source code directory as a volume in the dev container. [Docker](../Dockerfile) file contains all necessary environment setup to build and run DREAMPlace CUDA libraries and Cypress workflow. Make sure NVIDIA container toolchain is installed in host machine. Verify if NVIDIA drivers are visible and available inside the container by running `nvidia-smi` command inside the container. 

1. Build container image
```bash
docker compose build 
```
Run this command from source code directory 

2. Run the container in demon mode
```bash
docker compose up -d
```
3. Drop into dev-container shell 
```bash
docker exec -it <container-id> /bin/bash
```
4. Build DREAMPlace CPP librares 
- Create build directory
```bash
rm -rf build && mkdir build && cd build
```
NOTE: Above command triggers clean build
- Run cmake inside build directory
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=../
```
- After successful execution of cmake command, run make command
```bash
make -j$(nproc) && make install
```

5. Run Cypress using config file 
- Navigate to Cypress source code directory
- Run python script for Cypress component placement
```bash
python dreamplace/Placer.py configs/test_big.json 
```

## Steps to resolve common errors
1. ABI library link error
```bash
root@778eb851196f:/Cypress# python dreamplace/Placer.py configs/test_big.json 
Traceback (most recent call last):
  File "dreamplace/Placer.py", line 43, in <module>
    import dreamplace.NonLinearPlace as NonLinearPlace
  File "/Cypress/dreamplace/NonLinearPlace.py", line 39, in <module>
    import dreamplace.BasicPlace as BasicPlace
  File "/Cypress/dreamplace/BasicPlace.py", line 37, in <module>
    import dreamplace.ops.move_boundary.move_boundary as move_boundary
  File "/Cypress/dreamplace/ops/move_boundary/move_boundary.py", line 26, in <module>
    import dreamplace.ops.move_boundary.move_boundary_cpp as move_boundary_cpp
ImportError: /Cypress/dreamplace/ops/move_boundary/move_boundary_cpp.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN3c106detail14torchCheckFailEPKcS2_jRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```
Solution: 
- Remove all .so files from repo
```bash
find . -name "*.so" -type f -delete
```
- make sure CMAKE_CXX_ABI flag in CMakeLists matches with the python configuration in container
    ```bash
    python -c "import torch; print('PyTorch CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
    ```
    Using above command verify Pytorch ABI configuration in container. Make sure CMAKE_CXX_ABI flag matches with this
- build again

