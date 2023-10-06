# [Diffusion Optimization Models with Trajectory Alignment](https://arxiv.org/abs/2305.18470)


*Generative models have had a profound impact on vision and language, paving the way for a new era of multimodal generative applications. While these successes have inspired researchers to explore using generative models in science and engineering to accelerate the design process and reduce the reliance on iterative optimization, challenges remain. Specifically, engineering optimization methods based on physics still outperform generative models when dealing with constrained environments where data is scarce and precision is paramount. To address these challenges, we introduce Diffusion Optimization Models (DOM) and Trajectory Alignment (TA), a learning framework that demonstrates the efficacy of aligning the sampling trajectory of diffusion models with the optimization trajectory derived from traditional physics-based methods*.


<p align="center">
  <img src="https://github.com/georgosgeorgos/trajectory-alignment-diffusion/blob/main/assets/trajectory_matching_general.png" width="600px" alt="teaser" >
</p>

This repo contains code and experiments for:

> **Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation**   
> [Giorgio Giannone](https://georgosgeorgos.github.io/), [Akash Srivastava](https://akashgit.github.io), [Ole Winther](https://olewinther.github.io), [Faez Ahmed](https://meche.mit.edu/people/faculty/faez@MIT.EDU)  
> Conference on Neural Information Processing Systems (NeurIPS), 2023

[[paper](https://arxiv.org/abs/2305.18470)]
[[code](https://github.com/georgosgeorgos/trajectory-alignment-diffusion/)]

> **Diffusing the Optimal Topology: A Generative Optimization Approach**   
> [Giorgio Giannone](https://georgosgeorgos.github.io/), [Faez Ahmed](https://meche.mit.edu/people/faculty/faez@MIT.EDU)  
> International Design Engineering Technical Conferences (IDETC), 2023

[[paper](https://arxiv.org/abs/2303.09760)]
[[code](https://github.com/georgosgeorgos/trajectory-alignment-diffusion/)]



<p align="center">
    <img src="https://github.com/georgosgeorgos/trajectory-alignment-diffusion/blob/main/assets/hist_compliance_in_distro.png" width="400px" alt="teaser" >
    <img src="https://github.com/georgosgeorgos/trajectory-alignment-diffusion/blob/main/assets/consistency_benchmark_tm.png" width="400px" alt="teaser" >
</p>

<p align="center">
    <img src="https://github.com/georgosgeorgos/trajectory-alignment-diffusion/blob/main/assets/trajectory_alignment_search_matching.png" width="800px" alt="teaser" >
</p>


-------

## Installation

```bash
conda create -n dom python=3.8
conda activate dom

git clone https://github.com/georgosgeorgos/trajectory-alignment-diffusion/
cd trajectory-alignment-diffusion
```


The code has been tested on Ubuntu 20.04, Pytorch 1.13, and CUDA 11.6


---------
## Dataset

* 2d topology optimization dataset 64x64 - [here](https://github.com/georgosgeorgos/trajectory-alignment-diffusion/releases/download/2d-to-64x64/DATASET_TOPOLOGY_64_INTERMEDIATE_v1.zip)

<p align="center">
  <img src="https://github.com/georgosgeorgos/trajectory-alignment-diffusion/blob/main/assets/256_optimized.png" width="800px" alt="teaser" >
</p>

---------

## Evaluation

We use the benchmark provided in [TopoDiff](https://github.com/francoismaze/topodiff). Follow the instructions [here](https://github.com/francoismaze/topodiff#contents) to download the evaluation set.


----------

### Train the model

```bash

TRAIN_FLAGS="--batch_size 32 --save_interval 10000 --use_fp16 True"
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

DATA_FLAGS="--data_dir ./dom_dataset/"


CUDA_VISIBLE_DEVICES=0 \
python scripts/image_train_intermediate_kernel.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $DATA_FLAGS

```

---------

### Sample the model

```bash

#! /bin/sh
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3 --use_fp16 True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --timestep_respacing 100 --noise_schedule cosine"
DATA_FLAGS="--constraints_path ./dom_dataset/test_data/ --num_samples 1800"
CHECKPOINTS_FLAGS="--model_path ./dom_logdir/ema_0.9999_xxxxx.pt"


CUDA_VISIBLE_DEVICES=0 \
python scripts/sample_kernel_relaxation.py $MODEL_FLAGS $DIFFUSION_FLAGS $DATA_FLAGS $CHECKPOINTS_FLAGS

```

--------
## Acknowledgments

A lot of code and ideas were borrowed from:

* https://github.com/francoismaze/topodiff
* https://github.com/openai/improved-diffusion


## Citation

```bibtex
@article{giannone2023aligning,
  title={Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation},
  author={Giannone, Giorgio and Srivastava, Akash and Winther, Ole and Ahmed, Faez},
  journal={arXiv preprint arXiv:2305.18470},
  year={2023}
}
```

```bibtex
@article{giannone2023diffusing,
  title={Diffusing the optimal topology: A generative optimization approach},
  author={Giannone, Giorgio and Ahmed, Faez},
  journal={arXiv preprint arXiv:2303.09760},
  year={2023}
}
```
