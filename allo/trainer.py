import os
from typing import Dict
import numpy as np
import jax
from tqdm import tqdm
import wandb

from params import TrainArgs
from utils import Buffer
from .model import ALLO


def train_allo(allo: ALLO, buffer: Buffer, args: TrainArgs, ckpt_dir: str, key: jax.random.PRNGKey, checkpoint_dirs: Dict[int, str] = None) -> ALLO:
    """ALLO training function"""
    print(f"\nTraining ALLO for {args.allo_training_steps} steps")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sampling discount: {args.sampling_discount}")
    print(f"  Use discounted sampling: {args.use_discounted_sampling}")

    # determine checkpoint steps
    if checkpoint_dirs is not None:
        checkpoint_steps = sorted(checkpoint_dirs.keys())
        print(f"  Checkpoints at steps: {checkpoint_steps}")
    else:
        checkpoint_steps = []

    # compute normalization statistics
    allo.compute_stats(buffer)

    sampling_discount = args.sampling_discount if args.use_discounted_sampling else 0.0
    
    rng = np.random.default_rng(int(jax.random.randint(key, (1,), 0, 2**31 - 1)[0]))

    # training loop
    for step in tqdm(range(args.allo_training_steps), desc="Training ALLO"):
        # sample pairs from buffer
        batch_obs, batch_next_obs, batch_obs_2 = buffer.sample(batch_size=args.batch_size, discount=sampling_discount, rng=rng)
        
        # training step
        metrics = allo.train_step(batch_obs, batch_next_obs, batch_obs_2)
        
        # logging
        if step % 5000 == 0 and not args.debug:
            wandb.log({
                'step': step,
                'allo_loss': float(metrics['allo_loss']),
                'graph_loss': float(metrics['graph_loss']),
                'dual_loss_pos': float(metrics['dual_loss_pos']),
                'dual_loss_neg': float(metrics['dual_loss_neg']),
                'barrier_loss_pos': float(metrics['barrier_loss_pos']),
                'barrier_loss_neg': float(metrics['barrier_loss_neg']),
            })

        # save checkpoint if at checkpoint step
        current_step = step + 1
        if current_step in checkpoint_steps:
            ckpt_path = os.path.join(checkpoint_dirs[current_step], 'allo_model.pkl')
            allo.checkpoint(ckpt_path)
            print(f"\n  ALLO checkpoint saved at step {current_step} to {ckpt_path}")

    print("ALLO training completed!")

    # save final checkpoint
    if checkpoint_dirs is None:
        allo_save_path = os.path.join(ckpt_dir, 'allo_model.pkl')
        allo.checkpoint(allo_save_path)
        print(f"ALLO model saved to {allo_save_path}")

    return allo
