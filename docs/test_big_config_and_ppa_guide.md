# Cypress Configuration, Loss Functions, and Final PPA Guide

This document provides a comprehensive reference explaining all parameters defined in `configs/test_big.json`, the underlying loss functions of the Cypress PCB placement framework, and the "Final PPA" (Power, Performance, Area / Placement Quality Attributes) metrics reported in optimization logs such as `results_big/small-9/DREAMPlace.log`.

---

## 1. Overview of `configs/test_big.json`

The `configs/test_big.json` configuration file controls the execution of the GPU-accelerated global placement, grid binning, optimization schedules, and loss weights for PCB benchmark designs (such as `small-9`). Below is an exhaustive breakdown of every parameter in the configuration file.

### 1.1 Input & Execution Flags

| Parameter | Type | Value | Description |
| :--- | :--- | :--- | :--- |
| `aux_input` | `String` | `"benchmarks/small-9/small-9.aux"` | Path to the Bookshelf auxiliary (`.aux`) file that lists all input netlist files (`.nodes`, `.nets`, `.wts`, `.scl`, `.pl`) defining the PCB benchmark circuit. |
| `gpu` | `Integer` | `1` | Flag enabling GPU acceleration (1 = enabled, 0 = CPU-only). When enabled, PyTorch offloads tensor operations and custom CUDA kernels (e.g., wirelength, density, preconditioning) to an NVIDIA GPU. |
| `result_dir` | `String` | `"results_big"` | Target directory where output files (such as legalized placement `.gp.pl` files, logs, and iteration plot images) are saved. |
| `num_threads` | `Integer` | `8` | Number of CPU threads allocated for multi-threaded operations, data preprocessing, and CPU-bound routines like Abacus/Tetris legalization. |
| `global_place_flag` | `Integer` | `1` | Enables (1) or disables (0) the non-linear global placement optimization phase. |
| `legalize_flag` | `Integer` | `1` | Enables (1) or disables (0) the legalization phase (Tetris-like overlap removal followed by Abacus row-based legalization) executed after global placement. |
| `detailed_place_flag` | `Integer` | `0` | Enables (1) or disables (0) detailed placement optimization. Set to `0` in `test_big.json` to skip detailed placement. |
| `plot_flag` | `Integer` | `1` | Enables (1) or disables (0) the generation and saving of visual layout plots (PNG images) during optimization iterations. |

### 1.2 Grid & Density Settings

| Parameter | Type | Value | Description |
| :--- | :--- | :--- | :--- |
| `num_bins_x` | `Integer` | `256` | Number of horizontal bins in the placement grid. The board is discretized into this grid for calculating electrostatic potential energy and component density via 2D Fast Fourier Transform (FFT). |
| `num_bins_y` | `Integer` | `256` | Number of vertical bins in the placement grid for density and congestion mapping. |
| `target_density` | `Float` | `0.7` | The target packing density (70%). It sets the upper bound for component area occupancy per bin during global placement, ensuring components are sufficiently spread out to prevent severe overlaps and maintain routability. |

### 1.3 Global Placement Stages (`global_place_stages`)

A list defining sequential optimization stages. In `test_big.json`, there is one primary stage configured with the following parameters:

| Parameter | Type | Value | Description |
| :--- | :--- | :--- | :--- |
| `num_bins_x` / `num_bins_y` | `Integer` | `256` / `256` | Grid resolution for this stage. |
| `iteration` | `Integer` | `1000000` | Maximum iteration budget for global placement convergence. |
| `learning_rate` | `Float` | `0.01` | Initial learning rate for the gradient descent optimizer. |
| `wirelength` | `String` | `"weighted_average"` | Wirelength approximation model used (Weighted-Average model, differentiable relaxation of Half-Perimeter Wirelength). |
| `optimizer` | `String` | `"nesterov"` | Optimization algorithm used for position updates (Nesterov accelerated gradient descent). |
| `Llambda_density_weight_iteration` | `Integer` | `1` | Frequency / iteration step for updating the density weight $\lambda_D$. |
| `Lsub_iteration` | `Integer` | `1` | Number of inner optimization sub-iterations per outer step. |

### 1.4 Physics-Inspired Optimization Weights & Models

| Parameter | Type | Value | Description |
| :--- | :--- | :--- | :--- |
| `density_weight` | `Float` | `8e-5` | Initial Lagrange multiplier weight ($\lambda_D$) for the density penalty term in the loss function. This weight dynamically and exponentially increases over iterations to spread components and minimize overlap energy. |
| `gamma` | `Float` | `4.0` | Smoothing parameter $\gamma$ for the Log-Sum-Exp (LSE) / Weighted-Average wirelength model. It balances accuracy (approximating true HPWL more tightly at higher values) with mathematical differentiability and gradient smoothness. |
| `net_crossing_flag` | `Integer` | `1` | Enables (1) or disables (0) the net crossing minimization term in the objective function, crucial for reducing routing congestion on 2D PCB layers. |

### 1.5 Initialization, Constraints & Miscellaneous Parameters

| Parameter | Type | Value | Description |
| :--- | :--- | :--- | :--- |
| `random_seed` | `Integer` | `42` | Random seed ensuring reproducibility across stochastic initialization and random placement routines. |
| `scale_factor` | `Float` | `1.0` | Scaling factor applied to board coordinates and component dimensions during preprocessing. |
| `ignore_net_degree` | `Integer` | `100` | Threshold for net pin count. Nets connecting more than 100 pins (such as global power/ground nets) are excluded from wirelength and net-crossing optimization to prevent geometric distortion. |
| `enable_fillers` | `Integer` | `1` | Enables (1) filler cells during global placement to help smooth out the density distribution across sparse board regions. |
| `gp_noise_ratio` | `Float` | `0.025` (2.5%) | Random perturbation noise ratio added during initialization to break symmetry and prevent the optimization from getting trapped in symmetric local minima. |
| `stop_overflow` | `Float` | `0.001` (0.1%) | Convergence threshold for bin density overflow. When the overflow drops below 0.1%, global placement successfully terminates and proceeds to legalization. |
| `dtype` | `String` | `"float32"` | Floating-point data precision (`float32`) used for PyTorch tensors and CUDA computations, balancing precision and GPU memory/compute performance. |
| `random_center_init_flag` | `Integer` | `1` | Enables (1) random initialization of movable component coordinates clustered around the center of the PCB canvas. |
| `enable_rotation` | `Integer` | `1` | Enables (1) orientation-aware component rotation optimization using Gumbel-Softmax relaxation to find optimal discrete legal orientations ($\theta$). |

---

## 2. Loss Functions and Cost Formulation

Cypress models PCB placement as a multi-objective optimization problem solved via gradient descent. As detailed in the Cypress research paper, the combined global placement cost function (Equation 8) is defined as:

$$\mathcal{L}(x, y, \theta) = \text{WL}(x, y, \theta) + \lambda_D \mathcal{D}(x, y, \theta) + \lambda_{NC} \mathcal{NC}(x, y, \theta)$$

### 2.1 Wirelength Model ($\text{WL}$)
* **Purpose**: Minimizes total interconnect length between connected component pins.
* **Mathematical Formulation**: Cypress extends the differentiable Log-Sum-Exp (LSE) and Weighted-Average wirelength models to account for component dimensions and arbitrary legal rotations ($\theta$). By smoothing the non-differentiable Half-Perimeter Wirelength (HPWL), it allows gradient-based optimizers (like Nesterov) to efficiently tune component coordinates $(x, y)$ and orientation logits ($\theta$).

### 2.2 Density Model ($\mathcal{D}$)
* **Purpose**: Prevents component overlapping and enforces spacing by penalizing high local component concentration.
* **Mathematical Formulation**: Inspired by DREAMPlace, Cypress models components as electrostatic charges, density as potential energy, and density gradients as electric fields. Poisson's equation is solved across the bin grid using 2D Fast Fourier Transforms (FFT). 
* **Dynamic Schedule**: The weight $\lambda_D$ starts at `density_weight` (`8e-5`) and exponentially increases over iterations to progressively push overlapping components apart into a minimum-energy, non-overlapping state. Separate density maps are maintained for the top and bottom layers of double-sided PCBs.

### 2.3 Net Crossing Model ($\mathcal{NC}$)
* **Purpose**: Minimizes intersecting signal nets to enhance routability on constrained 2D PCB routing layers.
* **Mathematical Formulation**: Evaluates continuous net crossing metrics across the layout. The weight $\lambda_{NC}$ follows a linear schedule ($\lambda_{NC}^{t+1} = \lambda_{NC}^t + \eta$) to gradually suppress net crossings as components distribute across the board.

### 2.4 Combined Objective ($\text{Obj}$)
* **Definition**: The scalar objective value ($\mathcal{Obj}$) reported at each iteration in optimization logs (e.g., `Obj 1.287087E+37`) represents the evaluated sum of wirelength, weighted density penalty, and weighted net crossings. When numerical instability or extreme bin overcrowding occurs, $\mathcal{Obj}$ evaluates to `INF`.

---

## 3. "Final PPA" Parameters and Metrics (`DREAMPlace.log`)

At the conclusion of the placement run, DREAMPlace outputs a summary dictionary of **Final PPA** (Power, Performance, Area / Placement Quality Attributes) metrics. For example, in `results_big/small-9/DREAMPlace.log`:

```python
{'rsmt': inf, 'hpwl': inf, 'congestion': inf, 'density': inf, 'net_crossing': inf, 'iteration': 2721, 'objective': inf, 'overflow': 0.31517186760902405, 'max_density': 193.33868408203125}
```

Below is a tabular reference explaining each metric in this summary dictionary:

| PPA Metric | Data Type | Description & Behavior |
| :--- | :--- | :--- |
| `rsmt` | `Float` | Rectilinear Steiner Minimum Tree wirelength. Evaluates the tree-based interconnection length of nets. Reported as `inf` when placement diverges or fails. |
| `hpwl` | `Float` | Half-Perimeter Wirelength. The bounding-box wirelength summed across all nets. Reported as `inf` if coordinates become invalid (`NaN` or `INF`). |
| `congestion` | `Float` | Routing congestion metric evaluating estimated routing blockages. Reported as `inf` upon placement failure. |
| `density` | `Float` | Overall layout density evaluation metric. Reported as `inf` when overflow is excessive. |
| `net_crossing` | `Float` | Total evaluated net crossings across all nets in the final placement state. Reported as `inf` on failure. |
| `iteration` | `Integer` | The total number of iterations executed before termination (e.g., `2721` iterations in the `small-9` run). |
| `objective` | `Float` | The final scalar value of the loss function $\mathcal{L}$. Evaluates to `INF` if divergence occurs. |
| `overflow` | `Float` | The final density overflow ratio (e.g., `0.31517` or ~31.5% bin overflow). Measures the fraction of component area exceeding bin capacity. If overflow remains significantly above `stop_overflow` (`0.001` or 0.1%) when iterations complete, placement is deemed unsuccessful (`[WARNING] DREAMPlace - placement failed`). |
| `max_density` | `Float` | The maximum bin density ratio encountered across any bin on the board (e.g., `193.33` or 193.3% capacity). A value well above 1.0 indicates severe component overcrowding and local potential divergence, triggering warnings (`overflow is significant 0.315 or hpwl is infinity or nan, skip legalization and detail placement steps`) and skipping subsequent legalization and detailed placement phases. |
