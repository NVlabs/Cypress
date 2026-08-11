---
name: cypress-tuning
description: Systematically diagnoses and tunes Cypress GPU-accelerated PCB placement optimization runs. Use when analyzing DREAMPlace logs, fixing placement divergence or clumping, or tuning wirelength, density, rotation, and net crossing parameters.
---

# Cypress Placement Diagnostic & Hyperparameter Tuning Skill

This skill provides a systematic, cause-and-effect methodology to analyze telemetry, diagnose optimization regimes, and tune hyperparameter configurations for the **Cypress PCB Placement Framework**.

---

## 1. Diagnostic Decision Matrix

When evaluating placement runs, monitor `DREAMPlace.log` telemetry to identify the optimization regime and apply targeted hyperparameter adjustments:

```text
                  Observe Optimization Telemetry
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
 Regime A: Divergence     Regime B: Clumping    Regime C: Wire Crossings
(Gradient Explosion)    (Overflow Stalled)     (Poor Routability)
       │                       │                       │
       ▼                       ▼                       ▼
• γ ↑ (4.0 → 10.0)      • λ_D ↑ (8e-5 → 1e-3)   • enable_rotation: 1
• lr ↓ (0.01 → 0.002)   • bins ↑ (128 → 256)    • net_crossing_flag: 1
• net_crossing: 0       • target_density ↓      • γ ↓ in late stages
```

### Regime A: Gradient Explosion & Divergence
* **Observable Symptoms**: Log reports `Obj INF`, `NaN` gradients, HPWL exploding ($> 10^7$), or components pushed far outside board boundaries.
* **Root Cause**: Unstable gradient scales caused by dense net topology or steep net-crossing penalties dominating preconditioned updates.
* **Prescribed Adjustments**:
  1. **Increase Base Wirelength Scale ($\gamma$)**: Raise base `"gamma"` from `4.0` to `10.0` or `20.0` in the JSON config. A larger $\gamma$ distributes boundary forces smoothly across neighboring pins.
  2. **Disable Net Crossing Temporarily**: Set `"net_crossing_flag": 0` during initial stages to eliminate steep $O(N^2)$ Bézier segment penalties.
  3. **Reduce Learning Rate**: Lower `"learning_rate"` in `global_place_stages` from `0.01` to `0.002` or `0.005`.
  4. **Lower Initial Density Weight**: Set `"density_weight"` to `1e-5` to soften initial electrostatic repulsion forces.

### Regime B: Component Overlap & Stalled Overflow
* **Observable Symptoms**: Global placement terminates prematurely or stalls with high overflow ($\text{Overflow} > 0.1$), leaving cell clusters overlapping.
* **Root Cause**: Electrostatic density repulsion forces ($\nabla D$) are too weak relative to wirelength attraction forces ($\nabla WL$).
* **Prescribed Adjustments**:
  1. **Increase Initial Density Multiplier**: Boost `"density_weight"` from `8e-5` to `1e-3` or `5e-3`.
  2. **Refine FFT Grid Resolution**: Increase `"num_bins_x"` and `"num_bins_y"` (e.g., from `128` to `256`) in [`dreamplace/ops/electric_potential/electric_potential.py`](../../../dreamplace/ops/electric_potential/electric_potential.py).
  3. **Lower Target Density**: Reduce `"target_density"` from `0.8` to `0.6` or `0.5`, forcing the electrostatic field to push components into emptier regions.
  4. **Verify Filler Cells Enabled**: Ensure `"enable_fillers": 1` to guarantee smooth, continuous density field smearing during FFT spectral solves.

### Regime C: Excessive Net Crossings & Poor Routability
* **Observable Symptoms**: Low FreeRouting autorouter completion rate ($< 100\%$) or severe trace congestion despite low density overflow.
* **Root Cause**: Components placed in suboptimal orientations or on conflicting copper layer arrangements.
* **Prescribed Adjustments**:
  1. **Enable Rotation Optimization**: Set `"enable_rotation": 1` to execute Gumbel-Softmax continuous orientation relaxation ($\Theta = \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$) in [`dreamplace/ops/pin_pos/pin_pos.py`](../../../dreamplace/ops/pin_pos/pin_pos.py).
  2. **Enable Net Crossing Penalties**: Set `"net_crossing_flag": 1` to enforce $O(N^2)$ Bézier segment crossing penalties on identical copper layers via [`dreamplace/ops/net_crossing/net_crossing.py`](../../../dreamplace/ops/net_crossing/net_crossing.py).
  3. **Sharpen Boundary Forces in Late Stages**: Reduce `"gamma"` (e.g., $4.0 \to 1.0$) in later placement stages to force LSE/WA wirelength closer to exact HPWL bounding boxes.

### Regime D: Local Minima Trapping
* **Observable Symptoms**: Optimization converges rapidly but yields high HPWL and poor layout symmetry.
* **Root Cause**: Deterministic center initialization trapping the optimizer in a poor local minimum.
* **Prescribed Adjustments**:
  1. **Enable Center Randomization & Noise**: Set `"random_center_init_flag": 1` and increase `"gp_noise_ratio"` from `0.025` to `0.05` in [`dreamplace/BasicPlace.py`](../../../dreamplace/BasicPlace.py) to inject symmetry-breaking perturbations ($\mathcal{N}(0, \sigma^2)$).
  2. **Multi-GPU Bayesian Exploration**: Run MOTPE hyperparameter search (`./tuner/run_tuner.sh`) across parallel GPUs.

---

## 2. Telemetry Log Reference (`DREAMPlace.log`)

Cypress logs iteration metrics formatted by [`dreamplace/EvalMetrics.py`](../../../dreamplace/EvalMetrics.py):

```text
[INFO   ] DREAMPlace - iteration 2711, (2711,  0,  0), Obj 1.173403E+37, NetCrossing 1.805E+03, NetCrossingWeight 0.000000E+00, DensityWeight 5.399523E+34, HPWL 8.318736E+04, Overflow 7.420439E-03, MaxDensity 1.638E+00, gamma 6.443016E+00, time 7.787ms
```

| Field Name | Description & Grounding |
| :--- | :--- |
| `iteration` | Global placement step index (`NonLinearPlace.py`). |
| `(Lgamma, Llambda, Lsub)` | 3-nested loop stage indices: <br>• $L_\gamma$: Outer loop step index for wirelength parameter updates. <br>• $L_\lambda$: Middle loop step index for density multiplier updates. <br>• $L_{\text{sub}}$: Inner loop step index for Nesterov spatial coordinate descent. |
| `Obj` | Master loss $\mathcal{L} = WL + \lambda_D D + \lambda_{NC} NC$ evaluated in [`dreamplace/PlaceObj.py`](../../../dreamplace/PlaceObj.py). |
| `NetCrossing` | Pairwise segment crossing cost score evaluated via `net_crossing_op(pos)`. |
| `NetCrossingWeight` | Current multiplier weight $\lambda_{NC}$ for net crossing penalties. |
| `DensityWeight` | Current electrostatic density Lagrange multiplier $\lambda_D$. |
| `HPWL` | Exact Half-Perimeter Wirelength evaluated across all non-masked nets ($2 \le \text{degree} < 100$). |
| `Overflow` | Normalized density overflow ratio ($\text{Overflow} = \frac{\text{overflow\_area}}{\text{total\_movable\_node\_area}}$). Global placement terminates when $\text{Overflow} < \text{stop\_overflow}$ ($0.001$). |
| `MaxDensity` | Maximum local bin density occupancy ratio $\max(\rho_{\text{bin}})$ across all bins. |
| `gamma` | Runtime smoothness parameter $\gamma = \gamma_{\text{base}} \cdot 10^{\frac{20}{9}(\text{overflow} - 0.1) - 1}$. |
| `time` | Per-iteration forward/backward metric computation delay in milliseconds. |

---

## 3. Configuration Parameter Reference

Key JSON configuration fields (from `configs/test_big.json` or `dreamplace/params.json`):

```json
{
    "aux_input": "benchmarks/small-9/small-9.aux",
    "gpu": 1,
    "num_bins_x": 256,
    "num_bins_y": 256,
    "global_place_stages": [
        {
            "num_bins_x": 256,
            "num_bins_y": 256,
            "iteration": 1000000,
            "learning_rate": 0.01,
            "wirelength": "weighted_average",
            "optimizer": "nesterov"
        }
    ],
    "target_density": 0.7,
    "density_weight": 8e-5,
    "gamma": 4.0,
    "ignore_net_degree": 100,
    "enable_fillers": 1,
    "gp_noise_ratio": 0.025,
    "stop_overflow": 0.001,
    "random_center_init_flag": 1,
    "enable_rotation": 1,
    "net_crossing_flag": 1
}
```

---

## 4. Execution Commands

### Run Global Placement
```bash
uv run python dreamplace/Placer.py configs/your_config.json
```

### Run Hyperparameter Search (MOTPE)
```bash
./tuner/run_tuner.sh 1 1 test/tune/pcb-configspace.json /path/to/aux_file test/tuner/tuner-ppa.json "" 100 4 0 0 10 ./tuner ./results/result_path
```

### Output Coordinates Back-Annotation
Read `<design_name>.gp.pl` output file to extract final component center positions $(x_c, y_c)$ and legal orientation angles ($\theta_c \in \{0, \pi/2, \pi, 3\pi/2\}$).
