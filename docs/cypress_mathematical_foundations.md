# Cypress Internals: Mathematical Objective → Implementation → Optimization

## Purpose

This document captures the evolving understanding of the **Cypress component-placement framework** by mapping its implementation back to the mathematical concepts behind analytical placement.

The document is intentionally organized around the three major components of the Cypress objective:

1. **Wirelength**
2. **Density**
3. **Net Crossing**

This structure is intended to make the document extensible. Each objective component can be studied independently:

```text
Cypress Objective
│
├── 1. Wirelength
│     ├── HPWL
│     ├── LSE
│     ├── γ
│     ├── implementation
│     └── gradients
│
├── 2. Density
│     ├── placement bins
│     ├── electrostatic model (Poisson's equation)
│     ├── FFT spectral solution (DCT/DST)
│     ├── dual-side layer separation & orientation mapping
│     └── overflow, potential energy, and interaction with γ
│
└── 3. Net Crossing
      ├── pin-pair decomposition (Algorithm 1)
      ├── Bézier line segment formulation
      ├── smooth bell-shaped crossing penalty function
      └── layer-aware GPU CUDA parallelized implementation
```

The final goal is to establish:

```text
mathematical concept
        ↓
implementation mechanism
        ↓
observable optimization behavior
        ↓
hyperparameter relationship
        ↓
reproducible tuning procedure
        ↓
agent skill
```

---

# 1. Wirelength Objective

## 1.1 Why Wirelength Matters

A placement determines the physical positions of components and therefore the positions of their pins.

Components are connected by **nets**. A net represents electrical connectivity between a set of pins.

For one net, consider the horizontal pin coordinates:

```text
X = {x1, x2, x3, ...}
```

and vertical pin coordinates:

```text
Y = {y1, y2, y3, ...}
```

The pins define a minimum axis-aligned bounding box.

Its boundaries are:

- left: `xmin = min(X)`
- right: `xmax = max(X)`
- bottom: `ymin = min(Y)`
- top: `ymax = max(Y)`

The horizontal and vertical spans are:

$$
W = x_{\max}-x_{\min}
$$

$$
H = y_{\max}-y_{\min}
$$

The **Half-Perimeter Wirelength (HPWL)** is:

$$
HPWL =
(x_{\max}-x_{\min})
+
(y_{\max}-y_{\min})
$$

HPWL is a **wirelength proxy**, not the exact routed wire length.

The motivation is straightforward: reducing the bounding-box dimensions generally reduces the amount of routing required to connect the pins.

---

## 1.2 Why HPWL Is Difficult to Optimize Directly

The exact HPWL expression contains:

```text
max()
min()
```

For example:

$$
x_{\max}=\max(x_1,x_2,\ldots,x_n)
$$

The identity of the maximum can change as components move.

The resulting function is non-smooth and is not suitable for straightforward gradient-based optimization.

An analytical placer therefore replaces the hard extrema with differentiable approximations.

This is where **Log-Sum-Exp (LSE)** enters.

---

## 1.3 Log-Sum-Exp as a Smooth Maximum

For:

$$
\mathbf{x}=(x_1,x_2,\ldots,x_n)
$$

define:

$$
\operatorname{LSE}_{\gamma}(\mathbf{x})
=
\gamma
\ln
\left(
\sum_{i=1}^{n}
e^{x_i/\gamma}
\right)
$$

where:

$$
\gamma>0
$$

controls the smoothing scale.

As:

$$
\gamma\rightarrow0^+
$$

we obtain:

$$
\operatorname{LSE}_{\gamma}(\mathbf{x})
\rightarrow
\max(\mathbf{x})
$$

A useful bound is:

$$
x_{\max}
\le
\operatorname{LSE}_{\gamma}(\mathbf{x})
\le
x_{\max}+\gamma\ln n
$$

Therefore:

- LSE never underestimates the maximum.
- The worst-case overestimation is `γ ln(n)`.
- The worst case occurs when all values are equal.

For the minimum, the corresponding smooth approximation can be obtained by applying the same idea to the negated coordinates.

---

## 1.4 Relative-Gap Interpretation of γ

The worst-case `γ ln(n)` bound does not completely describe how LSE behaves on a real net.

Define:

$$
\Delta_i=x_{\max}-x_i
$$

Then:

$$
\operatorname{LSE}_{\gamma}(\mathbf{x})
=
x_{\max}
+
\gamma
\ln
\left(
1+
\sum_{i\ne\max}
e^{-\Delta_i/\gamma}
\right)
$$

This form exposes the actual mechanism.

Each non-maximum value contributes according to:

$$
e^{-\Delta_i/\gamma}
$$

Therefore:

- pins close to the maximum contribute strongly;
- pins far from the maximum are exponentially suppressed.

The important engineering interpretation is:

> **LSE primarily cares about pins that are close to the current extremum at the scale of γ.**

The total number of pins is therefore not the whole story.

---

## 1.5 Three Useful γ Regimes

### Clustered extrema

If several pins are close to the maximum:

$$
\Delta_i\approx0
$$

then:

$$
e^{-\Delta_i/\gamma}\approx1
$$

If `m` pins are effectively tied:

$$
\text{error}\approx\gamma\ln m
$$

Several pins therefore participate in defining the effective boundary.

---

### Isolated extremum

If the second-largest value is far from the maximum:

$$
\Delta_2\gg\gamma
$$

then:

$$
e^{-\Delta_2/\gamma}\approx0
$$

and LSE approaches the true maximum.

For example, a gap of approximately `5γ` suppresses an individual contribution to:

$$
e^{-5}\approx0.0067
$$

This is only an individual-term heuristic. With many pins, many small contributions can still accumulate.

The exact relative-gap expression remains the reliable description.

---

### Distributed values

When several values are moderately close to the maximum, LSE produces a graded gradient rather than selecting a single hard maximum.

This is one of the main reasons LSE is useful in gradient-based placement.

---

## 1.6 LSE Gradient: A Soft Boundary Force

The derivative of LSE is:

$$
\frac{\partial\operatorname{LSE}_{\gamma}}
{\partial x_i}
=
\frac{e^{x_i/\gamma}}
{\sum_j e^{x_j/\gamma}}
$$

Define:

$$
p_i=
\frac{e^{x_i/\gamma}}
{\sum_j e^{x_j/\gamma}}
$$

Then:

$$
\sum_i p_i=1
$$

This is the softmax distribution.

A pin close to the maximum receives a large share of the gradient.

A pin far from the maximum receives very little.

This can be viewed as a **soft bounding-box force field**.

The hard maximum effectively says:

> One pin defines the boundary.

LSE says:

> Several nearby pins may participate in defining the boundary, with exponentially decreasing influence.

---

## 1.7 Four Soft Boundaries for HPWL

HPWL has four extrema:

```text
              ymax
               ↑
               │
        ┌──────────────┐
 xmin ← │     net      │ → xmax
        └──────────────┘
               │
               ↓
              ymin
```

The implementation therefore needs smooth approximations for:

- `xmax`
- `xmin`
- `ymax`
- `ymin`

For the x direction, the resulting gradient has the conceptual form:

$$
\frac{\partial WL}{\partial x_i}
=
p_i^{\max}-p_i^{\min}
$$

The same principle applies to y.

---

## 1.8 Numerical Implementation of LSE

DREAMPlace/Cypress does not naively evaluate:

$$
\gamma\log\sum_i e^{x_i/\gamma}
$$

because direct exponentiation can overflow numerically.

Instead, it uses shifted exponentials.

For the maximum side:

$$
e^{(x_i-x_{\max})/\gamma}
$$

Since:

$$
x_i-x_{\max}\le0
$$

the exponential remains bounded.

For the minimum side:

$$
e^{(x_{\min}-x_i)/\gamma}
$$

The same transformation is applied independently in y.

This is an important connection between the mathematics and implementation:

> **The relative-gap formulation is closely reflected in the numerically stable form used by the actual wirelength operator.**

---

## 1.9 LSE Gradient in the Implementation

Conceptually, the implementation calculates:

$$
p_i^{\max}
=
\frac{
e^{(x_i-x_{\max})/\gamma}
}{
\sum_j e^{(x_j-x_{\max})/\gamma}
}
$$

and:

$$
p_i^{\min}
=
\frac{
e^{(x_{\min}-x_i)/\gamma}
}{
\sum_j e^{(x_{\min}-x_j)/\gamma}
}
$$

Then:

$$
\frac{\partial WL}{\partial x_i}
=
p_i^{\max}-p_i^{\min}
$$

There is therefore no explicit rule such as:

```text
if pin == xmax:
    apply force
```

The boundary behavior emerges from the exponential weighting.

---

## 1.10 Orientation-Aware Wirelength & Gumbel-Softmax Relaxation

Because PCB components can rotate to discrete legal orientations $\theta_c \in \Theta$ (e.g., $0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ}$), pin coordinates $(x_k, y_k)$ depend on component orientation:
$$
x_k = x_c + x_k^{\text{off}}\cos\theta_c - y_k^{\text{off}}\sin\theta_c, \quad
y_k = y_c + x_k^{\text{off}}\sin\theta_c + y_k^{\text{off}}\cos\theta_c
$$
Optimizing discrete orientations directly via gradient descent is infeasible. To make orientation optimization differentiable, Cypress relaxes the categorical choice of legal orientations using the **Gumbel-Softmax reparameterization trick** (`dreamplace/ops/pin_pos/pin_pos.py`). 

For each component, a categorical sample vector $\boldsymbol{\kappa}$ of dimension $n = |\Theta|$ is computed as:
$$
\kappa_i = \frac{\exp((\log(\pi_i) + g_i)/\tau)}{\sum_{j=1}^n \exp((\log(\pi_j) + g_j)/\tau)}, \quad i = 1, \dots, n
$$
where $g_i$ are i.i.d. samples drawn from $\operatorname{Gumbel}(0, 1)$ and $\tau$ is the softmax temperature. 

During the orientation optimization phase (Algorithm 2), the continuous approximation of orientation is expressed as:
$$
\hat{\theta}_c = \sum_{i=0}^n o_i \cdot \kappa_i
$$
where $o_i$ represents the legal orientation angles. This relaxation allows gradients to flow directly through the discrete orientation choices into the orientation logits $\boldsymbol{\pi}$, enabling joint bilevel optimization of spatial coordinates $(\mathbf{x}, \mathbf{y})$ and component orientations $\boldsymbol{\theta}$. Upon completion of global placement, the definitive legal orientation is selected via $k = \arg\max_i [g_i + \log(\pi_i)]$.

---

## 1.11 GPU Execution

The LSE wirelength operation has CPU and GPU implementations.

When placement tensors are on the GPU, the CUDA implementation is used.

Conceptually:

```text
Python placement objective
        ↓
LSE wirelength operator
        ↓
CUDA implementation
        ↓
wirelength value + gradient
```

This is important for the current Cypress work because GPU execution is part of the intended placement workflow.

---

## 1.11 γ in the DREAMPlace/Cypress Implementation

The user-facing `gamma` parameter should not be interpreted as a raw physical coordinate distance.

DREAMPlace scales the base γ using placement-bin dimensions.

Conceptually:

$$
\gamma_{\text{base}}
=
\text{params.gamma}
\left(
\text{bin\_size}_x+
\text{bin\_size}_y
\right)
$$

The runtime γ is then derived from this base value.

Therefore:

> **Changing placement resolution can indirectly change the physical scale of γ even when the configured `gamma` value remains unchanged.**

A γ value is meaningful only in the context of the placement geometry and bin configuration.

---

## 1.12 γ Is Coupled to Placement Dynamics

A critical implementation-level observation is that runtime γ is dynamically related to placement overflow.

Conceptually:

```text
high placement overflow
        ↓
larger γ
        ↓
smoother wirelength approximation

lower placement overflow
        ↓
smaller γ
        ↓
sharper wirelength approximation
```

Thus γ participates in a feedback loop:

```text
density
   ↓
overflow
   ↓
γ
   ↓
LSE wirelength gradient
   ↓
component movement
   ↓
density
```

Therefore:

> **γ should not be treated as an isolated wirelength parameter.**

---

## 1.13 Large γ vs Small γ

| Property | Large γ | Small γ |
|---|---|---|
| LSE approximation | Smoother | Closer to hard max/min |
| Gradient distribution | More distributed | More concentrated |
| Approximation error | Larger | Smaller |
| Boundary competition | More candidates participate | Fewer candidates dominate |
| Wirelength behavior | More exploratory/smooth | More precise/sharp |

It is safer to describe the difference in terms of smoothness and gradient concentration rather than making universal statements about curvature.

The exact curvature depends on both γ and the current softmax distribution.

---

# 2. Density Objective

## 2.1 Why Density Is Necessary

Minimizing wirelength alone can produce pathological placements where connected components collapse into single overlapping points.

The density objective introduces repulsive force fields:

> **Do not allow too much component area to accumulate in any local region.**

Therefore placement balances wirelength attraction against electrostatic density repulsion:

$$
F = WL + \lambda_D D(x, y, \boldsymbol{\theta})
$$

---

## 2.2 Electrostatic Analogy (ePlace / DREAMPlace)

Cypress models placement density analogously to an electrostatic system:
- **Components are electric charges**, with charge magnitudes proportional to their component areas (or weighted by movable pin counts / macro footprints).
- **Density distribution is electrostatic potential energy**, governed by Poisson's equation.
- **Density gradients are electric fields** that exert repulsive forces on components, pushing them out of congested bins.

---

## 2.3 Dual-Side PCB Density Modeling

Unlike 2D monolithic VLSI layouts, printed circuit boards support **dual-sided placement** (top and bottom copper layers).
Cypress extends the DREAMPlace electrostatic density model by maintaining **separate density maps for each side**:
1. **Orientation Mapping ($\boldsymbol{\theta}$)**: Components are first rotated according to their current orientation logits. Component rotation can swap width and height dimensions ($w \leftrightarrow h$), which directly alters how a component's footprint is mapped onto local placement bins (`electric_potential.py`).
2. **Layer Separation**: Components assigned to the top layer contribute exclusively to the top electrostatic density grid, while bottom-layer components contribute to the bottom grid. This models independent physical routing and placement capacities on each side of the board.

---

## 2.4 Spectral Solution via Fast Fourier Transform (FFT)

Solving Poisson's equation over thousands of placement bins directly in spatial domains is computationally prohibitive. 
Following ePlace and DREAMPlace, Cypress computes electric potential and electric field distributions efficiently via **Discrete Spectral Transforms (DCT/DST)** (`electric_potential.py`):
1. **Density Map Binning**: Component areas are smoothly smeared (binned) across adjacent placement bins using bell-shaped distribution functions (`ElectricDensityMapFunction`).
2. **Spectral Transformation**: The spatial density map is transformed into frequency domain coefficients ($a_{uv}$) via 2D Fast Fourier Transform (`dct2.forward`).
3. **Potential & Field Computation**: In the spectral domain, Poisson's equation is solved algebraically by dividing frequency coefficients by spatial frequency terms ($\omega_u^2 + \omega_v^2$):
   $$
   \phi_{uv} = \frac{a_{uv}}{\omega_u^2 + \omega_v^2}, \quad \xi_x = \frac{\omega_u a_{uv}}{\omega_u^2 + \omega_v^2}, \quad \xi_y = \frac{\omega_v a_{uv}}{\omega_u^2 + \omega_v^2}
   $$
4. **Inverse Spectral Transforms**: IDCT/DST (`idct_idxst`, `idct2`) yield the continuous **electric field maps ($\xi_x, \xi_y$)** and potential energy.
5. **GPU Acceleration**: Because FFT operations and tensor reductions execute efficiently on NVIDIA GPUs, density gradients for thousands of components are computed concurrently in milliseconds.

---

# 3. Net-Crossing Objective

## 3.1 Why Net Crossing Is Necessary for PCBs

While Half-Perimeter Wirelength (HPWL) minimizes bounding box spans, and density prevents overlapping components, neither metric captures **routing resource conflicts on the same copper layer**.

In VLSI designs, multilayer stacked routing allows thousands of vertical vias and dense multi-layer metal routing, where RUDY (Routing Congestion Estimator) models work well.
However, PCBs are constrained by limited routing layers (e.g., typically 2 copper layers in standard baseline evaluations) and flexible unconstrained routing. Two nets can have identical HPWL but completely different intersection profiles:

```text
Net A:  ────────╲
                 ╲
                  ╲──────
Net B:       ─────╱
```
The intersection of routing paths creates electrical shorts or unroutable congestion. Cypress therefore introduces a bespoke **net-crossing penalty ($NC$)** (`net_crossing.py`) to explicitly penalize intersecting net segments on the same layer.

---

## 3.2 Pin-Pair Decomposition and Bézier Line Segments

As detailed in Algorithm 1 of the paper, multi-pin nets are decomposed into collections of **source-sink pin pairs**:
- Each net $N_i = (\text{source}_i, \text{Sinks}_i)$ is broken down into line segments $L_1, L_2$ connecting the source pin to each individual sink pin.
- Each line segment $L$ is expressed in first-degree Bézier parametric form:
  $$
  L_1 = \begin{bmatrix} x_1 \\ y_1 \end{bmatrix} + t \begin{bmatrix} x_2 - x_1 \\ y_2 - y_1 \end{bmatrix}, \quad
  L_2 = \begin{bmatrix} x_3 \\ y_3 \end{bmatrix} + u \begin{bmatrix} x_4 - x_3 \\ y_4 - y_3 \end{bmatrix}
  $$
  where $t, u \in [0, 1]$ are Bézier parameters representing fractional positions along the line segments.

---

## 3.3 Smooth Bell-Shaped Crossing Penalty Function

To make segment intersections differentiable for gradient descent optimization, Cypress defines parametric intersection coordinates $(t, u)$ via algebraic ratios:
$$
t = \frac{(x_1 - x_3)(y_3 - y_4) - (y_1 - y_3)(x_3 - x_4)}{(x_1 - x_2)(y_3 - y_4) - (y_1 - y_2)(x_3 - x_4) + \epsilon}
$$
(with a small $\epsilon$ to prevent division by zero for parallel segments).
An intersection occurs when $0 \le t \le 1$ and $0 \le u \le 1$ simultaneously. 

### Intuitive & Mathematical Rationale for the Boundary Conditions ($0 \le t \le 1, 0 \le u \le 1$)

1. **Parametric Line Segment Domain**:
   - A line segment $L_1$ connecting source pin $P_1(x_1, y_1)$ to sink pin $P_2(x_2, y_2)$ is defined parametrically as:
     $$
     L_1(t) = P_1 + t(P_2 - P_1), \quad t \in \mathbb{R}
     $$
   - At $t = 0$, $L_1(0) = P_1$ (the source pin coordinate).
   - At $t = 1$, $L_1(1) = P_2$ (the sink pin coordinate).
   - For $0 \le t \le 1$, $L_1(t)$ traces points **strictly on the physical segment** between $P_1$ and $P_2$. Values $t < 0$ or $t > 1$ represent points on the infinite line extending beyond the physical segment boundaries.

2. **Infinite Lines vs. Finite PCB Traces**:
   - Solving $L_1(t) = L_2(u)$ determines the intersection point of the two **infinite lines** containing $L_1$ and $L_2$. In 2D space, any two non-parallel infinite lines will intersect somewhere.
   - However, physical PCB traces are **finite line segments** connecting pin pairs. A physical wire crossing occurs **if and only if** the intersection point falls within the physical span of *both* trace segments simultaneously.
   - Therefore, requiring $0 \le t \le 1$ AND $0 \le u \le 1$ mathematically guarantees that the intersection point resides on both physical net segments rather than on their imaginary extensions.

3. **Smooth Penalty Centering**:
   - The midpoint of segment $L_1$ occurs at $t = 0.5$. By shifting the input parameter to $(t - 0.5)$, the midpoint corresponds to 0.
   - For any valid physical segment point $t \in [0, 1]$, the shifted parameter $(t - 0.5)$ falls in $[-0.5, 0.5]$, where the bell function $B(t - 0.5)$ evaluates to positive values ($1 - \nu(t - 0.5)^2 > 0$).
   - For points far outside the physical segment ($|t - 0.5| \ge 1 \implies t \le -0.5$ or $t \ge 1.5$), $B(t - 0.5)$ smoothly evaluates to 0, preventing non-intersecting distant nets from generating spurious gradients.

To provide smooth gradients around intersection boundaries, Cypress applies a **bell-shaped penalty function $B(x)$**:
$$
B(x) = \begin{cases}
1 - \nu x^2, & 0 \le |x| \le 0.5 \\
\mu (|x| - \sigma)^2, & 0.5 \le |x| \le 1 \\
0, & |x| \ge 1
\end{cases}
$$
where $\nu, \mu, \sigma$ are tunable parameters satisfying continuity constraints. The pairwise segment crossing cost is formulated as:
$$
\text{Crossing}(L_1, L_2) = B(t - 0.5) B(u - 0.5)
$$
The total design net crossing cost is the sum over all pairwise segment intersections occurring **on the same copper layer**:
$$
NC = \sum_{(i, j) \text{ on same layer}} \text{Crossing}(L_i, L_j)
$$
Despite the $O(N^2)$ theoretical pair comparison complexity, this operation is highly parallelized across GPU CUDA cores (`net_crossing_cuda`), enabling real-time gradient propagation.

---

# 4. Total Cypress Objective

Once the three components are understood individually, they can be combined.

Conceptually:

$$
\mathcal{L} =
WL
+
\lambda_DD
+
\lambda_{NC}NC
$$

Therefore:

$$
\nabla \mathcal{L}
=
\nabla WL
+
\lambda_D\nabla D
+
\lambda_{NC}\nabla NC
$$

The optimization pipeline is then:

```text
                    component positions
                            │
                            ▼
                       pin positions
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
      Wirelength          Density        Net Crossing
          │                 │                 │
         HPWL           bin occupancy     net geometry
          │                 │                 │
         LSE             density model    crossing model
          │                 │                 │
          ▼                 ▼                 ▼
      ∇WL                ∇D               ∇NC
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                       total gradient
                            │
                            ▼
                      preconditioning
                            │
                            ▼
                         optimizer
                            │
                            ▼
                  updated component positions
                            │
                            └──────────► repeat
```

---

# 5. Optimization and Preconditioning

The raw objective gradient is not necessarily the gradient directly used to update component positions.

DREAMPlace/Cypress applies gradient preconditioning before the optimizer step.

Conceptually:

$$
\text{objective}
\rightarrow
\nabla \mathcal{L}
\rightarrow
\text{preconditioning}
\rightarrow
\text{optimizer}
\rightarrow
\text{new positions}
$$

Therefore:

> **The actual component trajectory is the result of the interaction between objective gradients, preconditioning, and optimizer dynamics.**

This is important when diagnosing divergence.

---

# 6. Why a Placement Can Look Good and Then Fail

A placement that looks promising for several hundred iterations and then fails should not automatically be diagnosed as a γ problem.

Possible interacting causes include:

```text
                     placement divergence
                             │
          ┌──────────────────┼──────────────────┐
          ↓                  ↓                  ↓
      γ dynamics       density dynamics    learning rate
          │                  │                  │
          └──────────────┬───┴──────────────────┘
                         ↓
                   gradient scale
                         │
                         ↓
                  preconditioning
                         │
                         ↓
                    optimizer
```

Cypress adds another branch:

```text
                     total objective
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
         HPWL           Density        Net Crossing
          │                │                │
          └────────────────┴────────────────┘
                           ↓
                         update
```

This is why diagnosing Cypress requires looking at the optimization trajectory rather than only the final placement.

---

# 7. Configuration Mapping & Experiment Telemetry

To enable reproducible tuning and diagnostic telemetry, we map the configuration parameters in `configs/test_big.json` directly to the underlying mathematical formulation and optimization dynamics of Cypress.

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
            "optimizer": "nesterov",
            "Llambda_density_weight_iteration": 1,
            "Lsub_iteration": 1
        }
    ],
    "target_density": 0.7,
    "density_weight": 8e-5,
    "gamma": 4.0,
    "random_seed": 42,
    "scale_factor": 1.0,
    "ignore_net_degree": 100,
    "enable_fillers": 1,
    "gp_noise_ratio": 0.025,
    "global_place_flag": 1,
    "legalize_flag": 1,
    "detailed_place_flag": 0,
    "stop_overflow": 0.001,
    "dtype": "float32",
    "plot_flag": 1,
    "random_center_init_flag": 1,
    "result_dir": "results_big",
    "num_threads": 8,
    "enable_rotation": 1,
    "net_crossing_flag": 1
}
```

## 7.1 Mapping `test_big.json` Parameters to Mathematical Concepts

| Configuration Parameter | Value in `test_big.json` | Mathematical & Algorithmic Mapping |
| :--- | :--- | :--- |
| **`aux_input`** | `"benchmarks/small-9/small-9.aux"` | **Design Database Ingestion**: Parsed via [`dreamplace/PlaceDB.py`](../dreamplace/PlaceDB.py) to read Bookshelf files (`.aux`, `.nodes`, `.nets`, `.wts`, `.pl`, `.scl`) establishing node sizes and pin relative offsets $(x_k^{\text{off}}, y_k^{\text{off}})$. |
| **`gpu`** | `1` | **Hardware Acceleration**: Checked in [`dreamplace/PlaceObj.py`](../dreamplace/PlaceObj.py) to enable PyTorch GPU CUDA tensor allocation and invoke compiled C++/CUDA operator kernels under [`dreamplace/ops/`](../dreamplace/ops/). |
| **`num_bins_x` / `num_bins_y`** | `256, 256` | **Poisson Grid Resolution**: Defines spatial bin dimensions $(w_{\text{bin}}, h_{\text{bin}})$ for 2D FFT ($256 \times 256$) in [`dreamplace/ops/electric_potential/electric_potential.py`](../dreamplace/ops/electric_potential/electric_potential.py). Dynamically scales base $\gamma$ via $\gamma_{\text{base}} = \text{gamma} \cdot (w_{\text{bin}} + h_{\text{bin}})$ in [`dreamplace/PlaceObj.py`](../dreamplace/PlaceObj.py). |
| **`gamma`** | `4.0` | **Wirelength Smoothing Scale**: Base coefficient scaling LSE/WA boundary approximation in [`dreamplace/PlaceObj.py`](../dreamplace/PlaceObj.py). Dynamically updated during placement via $\gamma = \gamma_{\text{base}} \cdot 10^{\frac{20}{9}(\text{overflow} - 0.1) - 1}$. |
| **`wirelength`** | `"weighted_average"` | **Differentiable Wirelength Model**: Selects Weighted-Average (WA) smooth approximation in [`dreamplace/ops/weighted_average_wirelength/weighted_average_wirelength.py`](../dreamplace/ops/weighted_average_wirelength/weighted_average_wirelength.py) over Log-Sum-Exp (LSE) for calculating bounding-box forces $\nabla WL$. |
| **`target_density`** | `0.7` | **Density Capacity Limit ($\rho_{\text{target}}$)**: Binds local bin occupancy to 70% in [`dreamplace/ops/electric_potential/electric_overflow.py`](../dreamplace/ops/electric_potential/electric_overflow.py). Governs the overflow objective: $\text{Overflow} = \sum \max(0, \rho_{\text{bin}} - \rho_{\text{target}})$. |
| **`density_weight`** | `8e-5` | **Initial Multiplier ($\lambda_D^{(0)}$)**: Initial Lagrange multiplier in [`dreamplace/PlaceObj.py`](../dreamplace/PlaceObj.py) weighting electrostatic potential energy $D(\mathbf{x}, \mathbf{y}, \boldsymbol{\theta})$ relative to wirelength. |
| **`net_crossing_flag`** | `1` | **Net-Crossing Penalty ($NC$)**: Enables $O(N^2)$ Bézier parametric segment intersection penalties $\lambda_{NC} NC(\mathbf{x}, \mathbf{y}, \boldsymbol{\theta})$ on identical copper layers via [`dreamplace/ops/net_crossing/net_crossing.py`](../dreamplace/ops/net_crossing/net_crossing.py). |
| **`enable_rotation`** | `1` | **Gumbel-Softmax Orientation Relaxation**: Configures orientation logits in [`dreamplace/PlaceDB.py`](../dreamplace/PlaceDB.py) and executes continuous orientation relaxation via [`dreamplace/ops/pin_pos/pin_pos.py`](../dreamplace/ops/pin_pos/pin_pos.py) during bilevel optimization loops in [`dreamplace/NonLinearPlace.py`](../dreamplace/NonLinearPlace.py). |
| **`optimizer` / `learning_rate`** | `"nesterov"`, `0.01` | **Spatial Nesterov Gradient Descent**: Step size $\eta = 0.01$ updating spatial coordinates $(x,y)$ with preconditioned gradients $\mathbf{P}_{x,y} \cdot \nabla \mathcal{L}$ in [`dreamplace/NesterovAcceleratedGradientOptimizer.py`](../dreamplace/NesterovAcceleratedGradientOptimizer.py). |
| **`stop_overflow`** | `0.001` | **Convergence Stopping Criterion**: Checked in [`dreamplace/NonLinearPlace.py`](../dreamplace/NonLinearPlace.py) and [`dreamplace/PlaceObj.py`](../dreamplace/PlaceObj.py); terminates global placement when total normalized density overflow falls below $0.1\%$ ($10^{-3}$). |
| **`ignore_net_degree`** | `100` | **Global Net Filtering**: Mask condition in [`dreamplace/BasicPlace.py`](../dreamplace/BasicPlace.py) ($2 \le \text{degree} < 100$) excludes nets with $\ge 100$ pins (e.g., VDD/GND power planes) from wirelength calculations. |
| **`random_center_init_flag` / `gp_noise_ratio`**| `1`, `0.025` | **Symmetry-Breaking Initialization**: Collapses initial component centers to board midpoint with $2.5\%$ Gaussian noise in [`dreamplace/BasicPlace.py`](../dreamplace/BasicPlace.py) and [`dreamplace/PlaceObj.py`](../dreamplace/PlaceObj.py) to escape poor local minima. |
| **`enable_fillers`** | `1` | **Continuous Density Smearing**: Places dummy filler blocks in empty bins via [`dreamplace/PlaceDB.py`](../dreamplace/PlaceDB.py) to maintain smooth density energy gradients during FFT spectral calculations in [`dreamplace/ops/electric_potential/electric_potential.py`](../dreamplace/ops/electric_potential/electric_potential.py). |
| **`legalize_flag`** | `1` | **Post-GP Legalization**: Snaps optimized continuous coordinates $(\mathbf{x}, \mathbf{y}, \boldsymbol{\theta})$ to non-overlapping discrete physical grid locations in [`dreamplace/NonLinearPlace.py`](../dreamplace/NonLinearPlace.py) via Tetris and Abacus algorithms. |

---

## 7.2 Observable Telemetry & Optimization Dynamics

For reproducible experiments and divergence diagnosis, Cypress telemetry prints detailed operational state variables to `DREAMPlace.log` at every iteration (formatted by [`dreamplace/EvalMetrics.py`](../dreamplace/EvalMetrics.py)):

```text
[INFO   ] DREAMPlace - iteration 2711, (2711,  0,  0), Obj 1.173403E+37, NetCrossing 1.805E+03, NetCrossingWeight 0.000000E+00, DensityWeight 5.399523E+34, HPWL 8.318736E+04, Overflow 7.420439E-03, MaxDensity 1.638E+00, gamma 6.443016E+00, time 7.787ms
```

### Breakdown of Logged Telemetry Fields & Code Base Grounding:

| Telemetry Metric | Code Field in `EvalMetrics.py` | Implementation & Derivation Grounding |
| :--- | :--- | :--- |
| **`iteration`** | `self.iteration` | Current step index in global placement optimization loop (`NonLinearPlace.py`). |
| **`(2711, 0, 0)`** | `self.detailed_step` | 3-nested loop stage indices $(L_\gamma, L_\lambda, L_{\text{sub}})$: <br>• $L_\gamma$: Outer loop index for wirelength parameter scaling. <br>• $L_\lambda$: Middle loop index for density multiplier updates. <br>• $L_{\text{sub}}$: Inner loop index for Nesterov coordinate descent. |
| **`Obj`** | `self.objective` | Master objective loss $\mathcal{L} = WL + \lambda_D D + \lambda_{NC} NC$ evaluated in `PlaceObj.py`. |
| **`NetCrossing`** | `self.net_crossing` | Pairwise segment crossing cost evaluated via `net_crossing_op(pos)` in `net_crossing.py`. |
| **`NetCrossingWeight`**| `self.net_crossing_weight` | Current multiplier $\lambda_{NC}$ weighting net crossing penalties in loss function. |
| **`DensityWeight`** | `self.density_weight` | Current Lagrange multiplier $\lambda_D$ weighting electrostatic potential energy $D(\mathbf{x}, \mathbf{y}, \boldsymbol{\theta})$. |
| **`HPWL`** | `self.hpwl` | Exact Half-Perimeter Wirelength evaluated across all non-masked nets ($2 \le \text{degree} < 100$). |
| **`Overflow`** | `self.overflow` | Normalized density overflow ratio: $\text{Overflow} = \frac{\text{overflow\_area}}{\text{total\_movable\_node\_area}}$ (`electric_overflow.py`). Reaching `< stop_overflow` ($0.001$) triggers GP completion. |
| **`MaxDensity`** | `self.max_density` | Maximum bin density occupancy ratio $\max(\rho_{\text{bin}})$ across all $256 \times 256$ floorplan bins. |
| **`gamma`** | `self.gamma` | Runtime smoothness parameter $\gamma$ dynamically adapted based on overflow: $\gamma = \gamma_{\text{base}} \cdot 10^{\frac{20}{9}(\text{overflow} - 0.1) - 1}$. |
| **`time`** | `self.eval_time` | Execution time for forward/backward metric evaluation step in milliseconds. |

### Secondary Telemetry & Log Markers:
- **Optimizer & Step Delays**: `optimizer step <ms>` and `full step <ms>` record CUDA kernel launch overhead and PyTorch backpropagation timing per iteration.
- **Orientation Selections**: `Best theta: tensor([...])` logs the final discrete orientation angles ($\theta_c \in \{0, \pi/2, \pi, 3\pi/2\}$) derived from continuous Gumbel-Softmax class probabilities ($\arg\max_i [g_i + \log(\pi_i)]$).
- **Divergence Warnings**: `[WARNING] DREAMPlace - overflow is significant ... or hpwl is infinity or nan` indicates numerical divergence, triggering early termination.

---

# 8. Synthesized Operational Guidelines & Tuning Playbook

With the mathematical formulation mapped directly to the Cypress engine implementation, engineers and automated agents can follow a systematic, cause-and-effect tuning playbook instead of relying on unconstrained trial-and-error.

```text
                  Observe Optimization Trametry
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

---

## 8.1 Diagnostic Playbook by Optimization Regime

### Regime A: Gradient Explosion & Divergence
* **Observable Symptoms**: Log reports `NaN` or exploding loss, HPWL shoots to infinity, or components are pushed outside board boundaries.
* **Root Cause**: Unstable gradient scales from dense net topology or steep net-crossing penalties dominating preconditioned updates.
* **Prescribed Adjustments**:
  1. **Increase Wirelength Smoothing ($\gamma$)**: Raise base `gamma` from `4.0` to `10.0` (or `20.0`) in the config. A larger $\gamma$ distributes boundary gradients across more pins, smoothing steep objective cliffs.
  2. **Disable Net Crossing Temporarily**: Set `"net_crossing_flag": 0` during initial stages to eliminate $O(N^2)$ Bézier segment penalty spikes.
  3. **Reduce Learning Rate**: Lower `"learning_rate"` in `global_place_stages` from `0.01` to `0.002` or `0.005`.
  4. **Reduce Initial Density Multiplier**: Set `"density_weight"` to `1e-5` to soften initial electrostatic repulsion.

---

### Regime B: Component Overlap & Stalled Overflow
* **Observable Symptoms**: Global placement terminates prematurely or stalls with high overflow ($> 0.1$), leaving cell clusters unseparated.
* **Root Cause**: Electrostatic density forces ($\nabla D$) are insufficient to overcome wirelength attraction ($\nabla WL$).
* **Prescribed Adjustments**:
  1. **Increase Initial Density Weight**: Boost `"density_weight"` from `8e-5` to `1e-3` or `5e-3`.
  2. **Refine FFT Grid Resolution**: Increase `"num_bins_x"` and `"num_bins_y"` (e.g., from `128` to `256`). Finer bins increase spatial density detection precision.
  3. **Lower Target Density**: Reduce `"target_density"` from `0.8` to `0.6` or `0.5`, forcing the electrostatic field to push components into emptier board regions.
  4. **Ensure Filler Cells Enabled**: Verify `"enable_fillers": 1` to guarantee smooth, continuous density gradient smearing during FFT spectral solves.

---

### Regime C: Excessive Net Crossings & Poor Routability
* **Observable Symptoms**: Low FreeRouting completion rate ($< 100\%$) or unroutable trace congestion despite zero density overflow.
* **Root Cause**: Components placed in suboptimal orientations or on conflicting copper layer arrangements.
* **Prescribed Adjustments**:
  1. **Enable Rotation Optimization**: Set `"enable_rotation": 1` to enable Gumbel-Softmax differentiable orientation relaxation ($\Theta = \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$).
  2. **Enable Net Crossing Penalties**: Set `"net_crossing_flag": 1` to enforce $O(N^2)$ Bézier segment crossing penalties on identical copper layers.
  3. **Sharpen Boundary Forces in Late Stages**: Lower `gamma` (e.g., from `4.0` to `1.0`) during final placement stages to force LSE wirelength closer to exact HPWL bounding boxes.

---

### Regime D: Suboptimal Local Minima Trapping
* **Observable Symptoms**: Optimization converges rapidly but yields high HPWL and poor component clustering.
* **Root Cause**: Deterministic center initialization trapping the optimizer in a poor local minimum.
* **Prescribed Adjustments**:
  1. **Enable Center Randomization & Noise**: Ensure `"random_center_init_flag": 1` and increase `"gp_noise_ratio"` from `0.025` to `0.05` to inject symmetry-breaking perturbations ($\mathcal{N}(0, \sigma^2)$).
  2. **Multi-GPU Bayesian Design Space Exploration**: Execute MOTPE search (`tuner/run_tuner.sh`) across parallel GPUs to discover Pareto-optimal parameters balancing HPWL, density, and net crossing.

---

## 8.2 Execution Protocol for Integration Engineers

When embedding Cypress into a custom EDA flow:

1. **Pre-Flight Ingestion**:
   Convert design files into Bookshelf format (`.aux`, `.nodes`, `.nets`, `.wts`, `.pl`, `.scl`). Ensure all coordinates and pin relative offsets $(x_k^{\text{off}}, y_k^{\text{off}})$ are scaled accurately in micrometers.
2. **Environment & C++ ABI Check**:
   Ensure PyTorch CUDA support is verified (`torch.cuda.is_available()`) and custom CUDA libraries (`_cuda.so`) are compiled with matching ABI flags (`-D_GLIBCXX_USE_CXX11_ABI=0`).
3. **Execution**:
   Run global placement via Python entry point:
   ```bash
   uv run python dreamplace/Placer.py configs/your_config.json
   ```
4. **Result Back-Annotation**:
   Parse the generated `<design_name>.gp.pl` placement file to extract final coordinates $(x_c, y_c)$ and orientation angles ($\theta_c$). Recursively shift all linked CAD sub-elements (pads, ports, silkscreens, traces) by the component displacement delta.

