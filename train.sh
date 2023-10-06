#! /bin/sh


TRAIN_FLAGS="--batch_size 32 --save_interval 10000 --use_fp16 True"
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

DATA_FLAGS="--data_dir ./dom_dataset/"


#CUDA_VISIBLE_DEVICES=2 \
python scripts/image_train_intermediate_kernel.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $DATA_FLAGS