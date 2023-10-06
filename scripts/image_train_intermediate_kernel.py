"""
Train the main diffusion model (regardless of guidance) on images.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from diffusion_optimization_model import dist_util, logger
from diffusion_optimization_model.dataset_intermediate_kernel import (
    load_data,
)  # we use the kernel relaxation + intermediate representations in this dataloader
from diffusion_optimization_model.resample import create_named_schedule_sampler
from diffusion_optimization_model.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from diffusion_optimization_model.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    # topology + (vf; x,y loads; x,y boundary_condition; kernels)
    args.in_channels = 1 + 1 + 4 + 4
    print(args)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        compliance_conditioning=args.compliance_conditioning,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        regressor=None,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        consistency_terms=args.consistency_terms,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        compliance_conditioning=True,
        consistency_terms=["consistency"],
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
