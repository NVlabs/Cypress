# Cypress Placement Experiment Findings

This document summarizes the findings and inferences from the experiment conducted to set up the Cypress placement tool and resolve divergence issues in the `power-circuit` benchmark.

## 1. Environment Setup and Build Process

### Findings:
- **Dependency Management**: The project uses `uv` for fast dependency management. While `requirements.txt` provides the base packages, C++ extensions are central to the performance of DREAMPlace (the core engine of Cypress).
- **C++ Extensions**: Building extensions like `pin_pos`, `electric_potential`, and `weighted_average_wirelength` requires a compatible C++ compiler and the Torch C++ API. 
- **Build Automation**: A custom build script or sequential execution of `python setup.py install` for each op in `dreamplace/ops` is necessary. We successfully automated this process to ensure all optimized kernels were available.

## 2. Blockage Implementation for Large Components

### Logic:
- Components with dimensions exceeding **7x7mm** (e.g., area > 49mm² for square components) are now automatically treated as **Fixed Terminals** (blockages).
- This is implemented at the **frontend** level in `cypress/converter.py` during the JSON-to-Bookshelf conversion.
- By marking them as `terminal` in the `.nodes` file, the placement engine naturally treats them as fixed obstacles that other components must avoid, and they are excluded from the movable optimization set.

### Inferences:
- **Constraint Hierarchies**: Implementing fixed terminal logic at the converter level is more robust than relying on the placement engine's internal macro detection, as it guarantees that these components will not be moved even during the early stages of global placement.
- **Dimensional vs. Area Threshold**: Initial attempts used a simple area-based threshold (Area > 49.0). However, we switched to a dimension-based check (`width > 7.0 and height > 7.0`) to avoid accidentally fixing long, thin components (like connectors or bus traces) that might still need some flexibility in placement. This ensures that only truly "bulky" components act as blockages.

## 3. Resolving Numerical Divergence (NAN/Infinity)

The `power-circuit` benchmark initially suffered from immediate divergence, producing `Obj NAN` and `HPWL -INF` within the first few iterations.

### Key Factors for Stability:
- **Numerical Dynamic Range**:
    - The original Bookshelf files used units that resulted in coordinates exceeding 100,000. 
    - In the Log-Sum-Exp (LSE) wirelength model, the term $exp(x/\gamma)$ is calculated. With $x \approx 10^5$ and $\gamma \approx 10^1$, the exponent is $10^4$, which far exceeds the capacity of `float32` ($\approx e^{88}$).
    - **Inference**: Numerical stability in GPU-accelerated placers is heavily dependent on the ratio of the coordinate range to the smoothing parameter $\gamma$.
- **Coordinate Scaling**:
    - We adjusted `scale_factor` to `0.001`. This transformed the coordinate space from [0, 100,000] to [0, 100].
    - This change alone significantly reduced the risk of overflow in the exponential terms.
- **Gamma Parameter Tuning**:
    - We increased `gamma` to `1,000,000`. 
    - **Inference**: While a smaller `gamma` provides a tighter approximation of the Half-Perimeter Wire Length (HPWL), it also makes the gradient more "peaky" and prone to overflow. A large `gamma` provides a smoother, more "global" gradient that is much more stable during the initial steps of the Nesterov optimizer.
- **Bin Grid Granularity**:
    - We reduced the density grid from `64x64` to `16x16`.
    - **Inference**: For benchmarks with a smaller number of components (like `power-circuit` with 162 movable cells), a coarse grid is often superior. A too-fine grid results in high-frequency noise in the density gradient, which can cause the Nesterov optimizer to oscillate and eventually diverge.

## 4. DREAMPlace Core Analysis

### Findings:
- **Macro Masking**: DREAMPlace has an internal mechanism in `dreamplace/PlaceDB.py` to identify movable macros based on a mean area threshold.
- **Density Accounting**: When movable macros are detected, the engine subtracts their area from the `total_movable_node_area` and adds it to `total_fixed_node_area` for density target calculations.
- **Inference**: This "automatic" macro detection is a fallback. For PCB design, explicit control over which components are fixed (via our `converter.py` logic) provides much more predictable results for the designer.

## 5. Conclusion and Recommendations

The Cypress tool is highly sensitive to the numerical distribution of the input data. To ensure successful placement on new PCB benchmarks, we recommend:
1. **Always verify the scale**: Aim for internal coordinates in the range [0, 1000].
2. **Start with high Gamma**: Begin with `gamma = 1e6` and gradually decrease if higher wirelength accuracy is needed.
3. **Adjust Bin Count to Component Count**: Roughly $\sqrt{N_{components}}$ is a good starting point for the number of bins per dimension.
4. **Pre-process Fixed Components**: Use the updated `is_fixed` logic in the converter to handle blockages before they reach the placement engine.

*This document was generated by an AI agent (OpenHands) on behalf of the user.*
