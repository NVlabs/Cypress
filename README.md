# Cypress: VLSI-Inspired Scalable GPU-Accelerated PCB Placement

# Dependency 

- [DREAMPlace](https://github.com/limbo018/DREAMPlace)
    - Commit b8f87eec1f4ddab3ad50bbd43cc5f4ccb0072892 
    - Other versions may also work, but not tested

- GPU architecture compatibility 6.0 or later (Optional)
    - Code has been tested on GPUs with compute compatibility 8.0 on DGX A100 machine. 

# Use Internal Dev Docker

### Find the container
- Join DL `Protect-NBU-Guest`
- Log in to NGC, go to `NV-Developer/nv-nbu` team
- In Private Registry/Containers, find Cypress container, or use this [link](https://registry.ngc.nvidia.com/orgs/b7z2uzy5hmfb/teams/nv-nbu/containers/cypress)
- Click Get Container, it shows the image tag, such as `nvcr.io/b7z2uzy5hmfb/nv-nbu/cypress:v1.1`

### Download the container
- On your developer server, first set up an NGC API key and log in to NGC, instructions are [here](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#nvcontainers).
- Then, user the image tag, run `docker pull <image_tag>`, such as `docker pull nvcr.io/b7z2uzy5hmfb/nv-nbu/cypress:v1.1`

### Start the container

- Run `docker run -ti --name <container_name> --gpus all <image_tag> /bin/bash`
- Such as: `docker run -ti --name cypress-dev --gpus all nvcr.io/b7z2uzy5hmfb/nv-nbu/cypress:v1.1 /bin/bash`
- Go to work directory: `cd /AutoDMP`

# How to Run PCB Placement

- PCB Placement requires a configuration json file, examples are available under `tests/pcb`.
- A configuration file provides all information needed to run a PCB placement job.
- To run placement: `python dreamplace/Placer.py <config.json>`, such as `python dreamplace/Placer.py test/pcb/p39.json`.

