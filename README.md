## Diffusion Optimization Models with Trajectory Alignment for Constrained Design Generation

Code for the paper `Aligning Optimization Trajectories with Diffusion Models for Constrained Design Generation`.


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

```