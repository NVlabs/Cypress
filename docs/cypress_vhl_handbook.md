# Cypress & VHL Integration Handbook

This document serves as a reference for developers working on the integration between Cypress (the placement engine) and the VHL (Virtual Hardware Loom) agent system. It covers the core conversion logic, configuration tuning, and deployment environment.

## 1. Bidirectional Converter (`cypress/converter.py`)

The `CircuitJSONConverter` is the primary bridge between `tscircuit` Circuit-JSON and the Bookshelf IR required by Cypress.

### Conversion Logic (JSON to Bookshelf)
*   **Normalization**: Coordinates are normalized such that the board occupies a positive quadrant starting at `(0,0)` in micrometer units (scaled by 1000 by default).
*   **Net Mapping**: Pins are mapped using **relative offsets** from the component center. This is crucial for Cypress to understand how moving a component affects wirelength.
*   **Row Generation**: `.scl` files are dynamically generated to cover the entire board area with 1-unit high placement rows.

### Back-annotation (Bookshelf to JSON)
When Cypress completes placement, the results must be reflected back into the original Circuit-JSON.
*   **Center Shifting**: The converter calculates the displacement (shift) of each component.
*   **Recursive Update**: Unlike simple placement tools, our converter recursively shifts **all associated objects** linked to a component ID. This includes:
    *   `pcb_smtpad` (SMD pads)
    *   `pcb_port` (Electrical ports)
    *   `pcb_silkscreen_path` (Outline drawings)
    *   `pcb_trace` (Vias and wire segments)
*   **Preservation**: Non-PCB elements (schematic, source metadata) are preserved exactly as they were.

## 2. Hyperparameter Guide

The Cypress engine uses a physics-inspired optimization model where components are "charged particles" that repel each other (density penalty) while being pulled by "springs" (wirelength penalty).

### 2.1 Grid & Density Settings
*   **`num_bins_x / num_bins_y`**: The board is discretized into this grid. Density is calculated per "bin." Higher values increase overlap detection precision but slow down computation.
*   **`target_density`**: The packing target (0.0 to 1.0). A value of `0.7` allows components to occupy up to 70% of any local area.
*   **`density_weight`**: The initial strength of the repulsion force. Increase this if components stay clumped together; decrease it if they "explode" off the board too fast.
*   **`stop_overflow`**: The stopping condition. Global placement terminates when total overlap (overflow) falls below this threshold (e.g., `0.001` = 0.1%).

### 2.2 Global Placement Stages (`global_place_stages`)
Placement is optimized in iterative stages. Each stage can tune:
*   **`iteration`**: Maximum steps for the stage. (e.g., 500 for rapid tests, 10,000+ for production).
*   **`learning_rate`**: The step size for the Nesterov optimizer. Typically `0.01`.
*   **`wirelength`**: The mathematical model for estimation. `"weighted_average"` is the standard smooth approximation.
*   **`optimizer`**: Usually `"nesterov"` (Nesterov's Accelerated Gradient).

### 2.3 Physical & Routing Constraints
*   **`gamma`**: A smoothing parameter for wirelength. Smaller values are more accurate but harder to optimize.
*   **`ignore_net_degree`**: Nets with more pins than this (e.g., VCC/GND) are ignored to prevent everything from being pulled into a single massive clump.
*   **`enable_fillers`**: Insert dummy blocks into empty spaces to create a continuous density gradient, helping the optimizer "push" real components more smoothly.
*   **`gp_noise_ratio`**: Adds a percentage of random noise to initial positions to prevent the optimizer from getting stuck in poor local minima.
*   **`enable_rotation`**: Enables gradient-based orientation optimization, allowing components to flip (0, 90, 180, 270 degrees).
*   **`net_crossing_flag`**: A Cypress-specific penalty for wires that cross each other, improving routability.

### 2.4 Control Flags
*   **`gpu`**: Enables NVIDIA GPU acceleration (`1` recommended).
*   **`random_center_init_flag`**: When `1`, ignores original centers and starts all components in the center with noise (best for fresh placements).
*   **`legalize_flag`**: Post-optimization step to snap components to a valid, non-overlapping grid coordinates.

## 3. Environment & Deployment

### UV Setup
For local development without Docker, use the `uv` tool to manage the virtual environment:
```bash
uv venv
uv pip install -r requirements.txt
# Rendering note: Requires cairocffi (system libcairo recommended)
```

### Docker Production Image (`Dockerfile.prod`)
The production environment uses a multi-path `LD_LIBRARY_PATH` to ensure PyTorch and custom Cypress `.so` libraries are discoverable by the linker.
*   **Entrypoint**: Defaults to `cypress/engine.py`.
*   **Workdir**: `/app`.

### How to run Cypress?
```bash
PYTHONPATH=$(pwd)/install uv run python dreamplace/Placer.py tmp_json.json
```
---
*This document is intended to be updated incrementally as new VHL-specific features are added.*
