# Cypress: VLSI-Inspired PCB Placement with GPU Acceleration - Code Walkthrough

This document provides a detailed walkthrough of the high-value component implementations in the Cypress repository, correlating the codebase with the methodologies presented in the research paper "Cypress: VLSI-Inspired PCB Placement with GPU Acceleration".

## 1. Bilevel Placement Optimization (Algorithm 2)
**Paper Reference:** Section 4.2.2 "Orientation-aware Wirelength Model" and Algorithm 2.

The paper proposes a bilevel optimization process where component coordinates (`x, y`) and component orientations (`θ`) are jointly optimized using gradient descent. 
In `dreamplace/NonLinearPlace.py`, this is implemented with two separate optimizers. 
- The primary optimizer (e.g., Nesterov) updates the spatial positions (`pos` and `position`).
- A secondary optimizer, `rot_optimizer` (Adam or SGD), updates the orientation variables `orient_logits`.

Around lines 402-427 in `NonLinearPlace.py`, we can observe the scheduled optimization of orientations:
```python
update_orient_cond = (iteration >= 500 and iteration % 100 == 0)
if params.enable_rotation and update_orient_cond:
    # Freezes positions and updates rotations
    model.freeze_pos = True
    obj_rot, grad_rot = model.obj_and_grad_fn(pos, orient_logits=self.orient_logits)
    rot_optimizer.step()
    ...
```

## 2. Orientation-aware Pin Positions and Gumbel-Softmax
**Paper Reference:** Section 4.2.2 "Orientation-aware Wirelength Model" (Equation 7).

To calculate wirelength with orientation awareness, the paper relaxes the categorical choice of legal orientations to a continuous approximation using the Gumbel-Softmax reparameterization trick. 
In `dreamplace/ops/pin_pos/pin_pos.py` (lines 173-178), this is directly implemented in the `PinPos` module:
```python
# rotation is enabled
y = torch.nn.functional.gumbel_softmax(self.orient_logits, tau=1.0, hard=True)
index_tensor = torch.arange(4).unsqueeze(0).expand_as(y).to(y.device)
choices = torch.sum(y * index_tensor, dim=1)
self.theta = choices * math.pi / 2
```
This enables the gradient to flow through the discrete orientation variables to intelligently update `orient_logits`.

## 3. Net Crossing Penalty
**Paper Reference:** Section 4.2.1 "Net Crossing".

To mitigate routing resource conflicts efficiently, the paper introduces a bespoke net crossing penalty based on line segment intersections using continuous Bézier approximations.
This calculation is computationally heavy ($O(N^2)$ for segments) and is highly paralleled on the GPU.
- The Python interface is defined in `dreamplace/ops/net_crossing/net_crossing.py`.
- The actual forward and backward passes leverage custom C++/CUDA extensions to execute rapidly across thousands of net-pin segments, computing intersection points and adding standard distance penalties over overlapping routing segments on the same layer.

## 4. Constraint Handling & Fixed Components
**Paper Reference:** Section 4.5.1 "Spacing Constraints" and 4.5.2 "Fixed Components".

Spacing constraints (macro halos) and blockages are integrated iteratively. Fixed components simply have their update gradients masked out or are stored in constant tensors which are fed into the density/wirelength objective calculations but ignored during the coordinate updates. 
In `dreamplace/NonLinearPlace.py`, `self.op_collections.move_boundary_op(pos)` directly forces nodes to stay inside the routing boundary, satisfying geometrical limits.

## 5. Density and Poisson’s Equation
**Paper Reference:** Section 4.2.3 "Density".

Similar to DREAMPlace, Cypress models component density analogously to an electrostatic system where cells are charges.
The multi-layer aspect of PCBs requires independent density mapping fields for top and bottom sides. This logic relies on operations such as `dreamplace/ops/electric_potential/` and updates density effectively scaling forces depending on the node's dimensions—which in turn depend dynamically on its continuously modeled orientation.

## 6. Optimization Environment and Divergence Prevention
**Paper Reference:** Section 4.3 "Preconditioning".

Given the heavily varied bounds of components (ICs vs 0402 resistors), divergence-aware preconditioning adjusts the gradient step scale inversely proportional to node pin counts and bounding areas. In `NonLinearPlace.py` (Lines 495-540), divergence heuristics like `check_divergence` are specifically calibrated to prevent runaway gradient values that explode quickly and ruin global optimization, a critical adaptation to PCB component distributions.


## How to run the code in container?

Following is the history from the successful container
```bash
root@95e4001e10d0:/app# history
    1  pip install -r requirements.txt 
    2  python dreamplace/Placer.py test_small1.json
    3  rm build
    4  rm -rf build
    5  python dreamplace/Placer.py test_small1.json
    6  cd /app
    7  rm -rf build
    8  find dreamplace -name "*.so" -delete
    9  mkdir build && cd build
   10  cmake ..   -DCMAKE_INSTALL_PREFIX=/app   -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"   -DCMAKE_CUDA_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"   -DCMAKE_BUILD_TYPE=Release   -DUSE_CUDA=TRUE
   11  nvidi-smi
   12  python -c "import torch; print(torch.cuda.is_available())"
   13  make -j$(nproc)
   14  make install 
   15  python - <<EOF
import dreamplace.configure as cfg
print(cfg.compile_configurations)
EOF

   16  python dreamplace/Placer.py test_small1.json
   17  cd ../
   18  python dreamplace/Placer.py test_small1.json
   19  ls
   20  ls results/
   21  python dreamplace/Placer.py test_big.json 
   22  ls results
   23  clear
   24  ls
   25  rm test_big.json 
   26  python dreamplace/Placer.py test_big.json 
   27  rm test_big.json 
   28  ls results
   29  rm results_big/
   30  rm -rf results_big/
   31  clear
   32  python dreamplace/Placer.py test_big.json 
   33  ls results_big/
   34  ls results_big/small-9/
   35  history
```
