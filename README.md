# Cypress: VLSI-Inspired Scalable GPU-Accelerated PCB Placement

## Requirement

- GPU architecture compatibility 6.0 or later (Optional)
    - Code has been tested on GPUs with compute compatibility 8.0 on DGX A100 machine. 

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
./tuner/run_tuner.sh 1 1 	test/tune/pcb-configspace.json 	/path/to/aux_file 	test/pcb/tuner-ppa.json "" 	100 4 0 0 10 	./tuner 	./results/result_path
```