# Repository Context & Pipeline Integration Guide: Cypress

## 1. Overview & Architecture

**Cypress** is a scalable, GPU-accelerated PCB placement framework inspired by VLSI physical design methodologies (adapted from the **DREAMPlace** framework). It models component placement as a continuous non-linear optimization problem, treating components as charged particles with density repulsion forces and wirelength "springs," augmented with rotation optimization (via Gumbel-Softmax) and net-crossing penalties for routability.

### Dual-Phase Pipeline
1. **Data Preparation & Translation (`pcb-util/`)**: Ingests raw PCB designs (KiCad, IDF, or custom JSON) and converts them into an Intermediate Representation (YAML IR) and subsequently into the VLSI standard **Bookshelf** format (`.aux`, `.nodes`, `.nets`, `.wts`, `.pl`, `.scl`).
2. **Global Placement & Optimization (`dreamplace/`)**: A PyTorch-based, GPU-accelerated engine that executes multi-stage non-linear gradient descent optimization (`NonLinearPlace.py`) to minimize wirelength, density overflow, and net crossings while optimizing component orientations.

---

## 2. Repository Structure & Key Modules

- **`dreamplace/`**: Core placement engine.
  - `Placer.py`: Primary execution entry point.
  - `NonLinearPlace.py`: Main optimization loop, optimizer coordination (Nesterov for positions, Adam/SGD for orientation logits), and divergence checks.
  - `PlaceDB.py`: Python wrapper around the C++ design database.
- **`dreamplace/ops/`**: High-performance C++ and CUDA extension operators (compiled into `.so` shared libraries):
  - `wirelength`: LSE (Log-Sum-Exp) and Weighted-Average wirelength models.
  - `electric_potential`: Fast Fourier Transform (FFT)-based density electrostatics.
  - `net_crossing`: Bespoke $O(N^2)$ segment intersection penalty for routability.
  - `pin_pos`: Orientation-aware pin coordinate calculations using Gumbel-Softmax relaxation.
  - `move_boundary`: Boundary constraint enforcement.
- **`pcb-util/`**: PCB parsers, format converters, and visualization utilities.
- **`configs/`**: JSON configuration parameter sets (e.g., `test_small1.json`, `test_big.json`).
- **`docs/`**: Detailed technical guides (`cypress_workflow_documentation.md`, `cypress_vhl_handbook.md`, `gpu_build_guide.md`, `walkthrough.md`, `pcb_ir_spec.md`).
- **`tuner/`**: Hyperparameter search and PPA (Power, Performance, Area) tuning scripts.
- **`scripts/`**: Utility scripts, combinatorial Gumbel-Softmax tests, and flow runners.

---

## 3. Build, Compilation & Environment Setup

> **Important**: Build and run Cypress **inside the Docker container**. Do NOT attempt to build or run Cypress on the host machine or set up `uv`/python environments on the host, as underlying libraries and modules are outdated. The workspace is mounted as a volume into the devcontainer, allowing you to edit code and collect artifacts from the host machine.

### 3.1. Docker Container Setup
Ensure the NVIDIA Container Toolkit is installed on the host machine.

1. **Build container image** (from source root):
   ```bash
   docker compose build
   ```
2. **Run container in daemon mode**:
   ```bash
   docker compose up -d
   ```
3. **Enter dev-container shell**:
   ```bash
   docker exec -it <container-id> /bin/bash
   ```
4. **Verify GPU visibility**:
   ```bash
   nvidia-smi
   ```

### 3.2. Compiling C++/CUDA Operators (Inside Container)
Inside the container environment:

1. **Clean build directory**:
   ```bash
   rm -rf build && mkdir build && cd build
   ```
2. **Run CMake configuration**:
   ```bash
   cmake .. -DCMAKE_INSTALL_PREFIX=../
   ```
3. **Compile and install libraries**:
   ```bash
   make -j$(nproc) && make install
   ```
   This compiles the CUDA kernels and installs `.so` libraries into `dreamplace/ops/`.

---

## 4. Pipeline Integration Guide for Engineers

When integrating Cypress into a custom CAD/EDA pipeline, follow these 4 operational steps:

### Step 1: Ingestion & Format Translation
- Translate your custom PCB design representation (e.g., Circuit-JSON, KiCad, IDF) into the **Bookshelf** format (`.aux`, `.nodes`, `.nets`, `.wts`, `.pl`, `.scl`) using `pcb-util/` converters or custom scripts.
- Ensure the board dimensions, placement rows (`.scl`), and pin relative offsets are correctly normalized into micrometer or standard database units.

### Step 2: Configuration Parameterization
- Create or modify a JSON configuration file (refer to `configs/test_small1.json` or `dreamplace/params.json`).
- **Critical Configuration Fields**:
  - `"aux_input"`: Path to the target `.aux` Bookshelf file.
  - `"gpu"`: Set to `1` for GPU acceleration, `0` for CPU.
  - `"target_density"`: Packing density limit (e.g., `0.7` to `0.9`).
  - `"global_place_stages"`: Array defining iteration budgets, learning rates, and bin grid dimensions (`num_bins_x`, `num_bins_y`).
  - `"enable_rotation"`: Set to `true` to enable gradient-based component orientation optimization (0, 90, 180, 270 degrees).
  - `"net_crossing_flag"`: Enable/disable routability net crossing penalties.
  - `"result_dir"`: Output directory for placement artifacts.

### Step 3: Execution
Run the placement engine inside the container:
```bash
python dreamplace/Placer.py configs/test_big.json
```

### Step 4: Result Extraction & Back-Annotation
- Cypress outputs final global placement coordinates to `<design_name>.gp.pl`.
- Parse the `.gp.pl` file to extract updated component center coordinates $(x, y)$ and rotations $(\theta)$.
- **Back-Annotation Note**: If updating downstream CAD formats (like Circuit-JSON), apply center coordinate displacement shifts recursively to all associated sub-elements (SMD pads, ports, silkscreen paths, traces/vias) as detailed in `docs/cypress_vhl_handbook.md`.

---

## 5. Extending & Customizing Cypress

- **Adding Custom Cost Functions / Constraints**: Modify `dreamplace/NonLinearPlace.py`. Specifically, update `one_descent_step()` or gradient aggregation logic to incorporate thermal, height, or signal-integrity constraints into the loss function $\mathcal{L}$.
- **Developing Custom Operators**: Implement new operators in C++/CUDA under `dreamplace/ops/` and bind them to PyTorch via `setup.py` / `CMakeLists.txt`.

---

## 6. Known Issues, Divergence & Troubleshooting

- **Benchmark Divergence (`power-circuit`, etc.)**: Complex or dense layouts may experience gradient divergence. Resolve by increasing `gamma` (e.g., to `10.0`) and disabling `net_crossing_flag` in the JSON config.
- **ABI Mismatches / Link Errors**: If importing operator shared libraries fails with undefined symbol errors (e.g. `_ZN3c106detail14torchCheckFail...`):
  1. Remove all compiled `.so` files:
     ```bash
     find . -name "*.so" -type f -delete
     ```
  2. Check PyTorch CXX11 ABI configuration in the container:
     ```bash
     python -c "import torch; print('PyTorch CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
     ```
  3. Ensure `CMAKE_CXX_ABI` in `CMakeLists.txt` matches the PyTorch ABI flag, then clean and rebuild (`rm -rf build && mkdir build ...`).
- **Visual Debugging & Monitoring**:
  - Enable step snapshots by setting `"plot_flag": 1`, `"plot_frequency": 100`, and `"create_gif": 1` in the JSON config.
  - Monitor logs for HPWL degradation, overflow reduction rates, and divergence warnings in `DREAMPlace.log`.
