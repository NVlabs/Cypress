# Cypress: VLSI-Inspired Scalable GPU-Accelerated PCB Placement

Cypress is a scalable, GPU-accelerated PCB placement method inspired by VLSI. It incorporates
tailored cost functions, constraint handling, and optimized techniques adapted for PCB layouts.

<p align="center">
  <img src="images/big-3.gif" width="60%" alt="Cypress PCB Placement Animation">
</p>


## How to Build

You can build in two ways:

- Build without Docker by following the instructions of the DREAMPlace build at README_DREAMPlace.md.
- Use the provided Dockerfile to build an image with the required library dependencies.

## Run PCB Placement

```sh
cd /Cypress
python dreamplace/Placer.py path/to/your/config.json
```

## Run Hyperparameter search

```sh
./tuner/run_tuner.sh 1 1 	test/tune/pcb-configspace.json 	/path/to/aux_file 	test/tuner/tuner-ppa.json "" 	100 4 0 0 10 	./tuner 	./results/result_path
```

## Acknowledgement

Cypress builds upon and extends the following open-source projects:

- [DREAMPlace](https://github.com/limbo018/DREAMPlace): A deep learning toolkit-enabled VLSI placement framework that provides the foundation for our GPU-accelerated placement engine.

- [AutoDMP](https://github.com/NVlabs/AutoDMP): An automated design methodology platform that inspired aspects of our hyperparameter tuning approach.

We thank the authors and contributors of these projects for their valuable work that made Cypress possible.

## Citation

If you find this work useful, please cite the following paper:

```bibtex
@inproceedings{zhang2025cypress,
  title={Cypress: VLSI-Inspired PCB Placement with GPU Acceleration},
  author={Zhang, Niansong and Agnesina, Anthony and Shbat, Noor and Leader, Yuval and Zhang, Zhiru and Ren, Haoxing},
  booktitle={Proceedings of the 2025 International Symposium on Physical Design},
  pages={xx--xx},
  year={2025}
}
```