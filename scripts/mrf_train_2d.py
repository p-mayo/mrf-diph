"""
Train the MRF model with Deli-CS data.
"""
import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.mrf_image_datasets_2d import load_data as load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    mrf_model_and_diffusion_defaults,
    mrf_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    logger.log('Parameters:')
    params_str = ''
    for k, v in args_to_dict(args, mrf_model_and_diffusion_defaults().keys()).items():
        params_str += f'\t{k.upper()}: {v}\n'
    logger.log(params_str)
    logger.log("creating model...")
    model, diffusion = mrf_create_model_and_diffusion(
        **args_to_dict(args, mrf_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    data = load_data(
        image_size=args.image_size,
        input_dir=args.input_dir,
        ref_dir=args.ref_dir,
        batch_size=args.batch_size,
        vols=range(1, 11),
        slices=range(30),
        cut=args.cut,
        use_condition=args.use_condition,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
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
    ).run_loop()

def create_argparser():
    defaults = dict(
        input_dir='',
        ref_dir='',
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        ksvd=5,
        cut=3,
        use_condition=True,
        dims=2,
    )
    defaults.update(mrf_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
